"""Measure dense and anchor-chunked SDPA including mask construction and backward.

Example on Ascend:
    python -m specforge.benchmarks.benchmark_dflash_attention --device npu \
        --dtype bfloat16 --context-length 8192 --num-anchors 512 --block-size 16 \
        --batch-size 2 --heads 32 --kv-heads 8 --head-dim 128 --chunk-blocks 16 32 64
"""

import argparse
import importlib
import json
import statistics
import time

import torch
from torch.nn import functional as F

from specforge.modeling.draft.anchor_attention import (
    AnchorBlockMask,
    anchor_block_attention,
)
from specforge.modeling.draft.dflash_mask import create_dflash_sdpa_mask


def measure(args, tensors, anchors, keep, probe, chunk):
    device = tensors[0].device
    accelerator = None if device.type == "cpu" else getattr(torch, device.type)

    def synchronize():
        if accelerator is not None:
            accelerator.synchronize(device)

    def step():
        for tensor in tensors:
            tensor.grad = None
        if chunk:
            mask = AnchorBlockMask(
                anchors,
                keep,
                args.context_length,
                args.block_size,
                chunk,
                trim_context=args.trim_context,
            )
            output = anchor_block_attention(
                *tensors,
                mask,
                is_causal=args.causal,
                sliding_window=args.window,
                scale=args.head_dim**-0.5,
            )
        else:
            mask = create_dflash_sdpa_mask(
                anchors,
                keep,
                args.context_length,
                args.block_size,
                device,
                is_causal=args.causal,
                sliding_window=args.window,
            )
            output = F.scaled_dot_product_attention(
                *tensors,
                attn_mask=mask,
                scale=args.head_dim**-0.5,
                enable_gqa=args.heads != args.kv_heads,
            )
        (output.float() * probe).sum().backward()

    for _ in range(args.warmup):
        step()
    for tensor in tensors:
        tensor.grad = None
    synchronize()
    resident = None
    if accelerator is not None:
        resident = accelerator.memory_allocated(device)
        accelerator.reset_peak_memory_stats(device)
    durations = []
    for _ in range(args.iterations):
        synchronize()
        started = time.perf_counter()
        step()
        synchronize()
        durations.append((time.perf_counter() - started) * 1000)
    peak = None if accelerator is None else accelerator.max_memory_allocated(device)
    return dict(
        attention_chunk_blocks=chunk,
        attention_trim_context=bool(chunk and args.trim_context),
        median_forward_backward_ms=statistics.median(durations),
        peak_allocated_bytes=peak,
        peak_above_resident_bytes=None if peak is None else peak - resident,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype", choices=("float32", "bfloat16", "float16"), default="float32"
    )
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--num-anchors", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--chunk-blocks", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--causal", action="store_true")
    parser.add_argument(
        "--trim-context",
        action="store_true",
        help="Resolve each tile's context prefix once on the host; timing includes the D2H boundary.",
    )
    parser.add_argument("--window", type=int)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    for name in (
        "context_length",
        "num_anchors",
        "block_size",
        "batch_size",
        "heads",
        "kv_heads",
        "head_dim",
        "iterations",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmup < 0 or any(chunk < 1 for chunk in args.chunk_blocks):
        parser.error("warmup must be nonnegative and chunk sizes must be positive")
    if args.heads % args.kv_heads or (args.window is not None and args.window < 1):
        parser.error("heads must be divisible by kv-heads and window must be positive")
    if args.device.split(":")[0] == "npu":
        importlib.import_module("torch_npu")
    device, dtype = torch.device(args.device), getattr(torch, args.dtype)
    if device.type != "cpu":
        getattr(torch, device.type).set_device(device)
    torch.manual_seed(42)
    queries = args.num_anchors * args.block_size
    anchors = torch.linspace(
        0, args.context_length - 1, args.num_anchors, device=device
    ).long()
    anchors = anchors[None].expand(args.batch_size, -1).contiguous()
    keep = torch.ones_like(anchors, dtype=torch.bool)
    tensors = tuple(
        torch.randn(*shape, device=device, dtype=dtype, requires_grad=True)
        for shape in (
            (args.batch_size, args.heads, queries, args.head_dim),
            (
                args.batch_size,
                args.kv_heads,
                args.context_length + queries,
                args.head_dim,
            ),
            (
                args.batch_size,
                args.kv_heads,
                args.context_length + queries,
                args.head_dim,
            ),
        )
    )
    probe = torch.randn(
        args.batch_size, args.heads, queries, args.head_dim, device=device
    )
    results = []
    for chunk in (0, *args.chunk_blocks):
        try:
            result = measure(args, tensors, anchors, keep, probe, chunk)
        except torch.OutOfMemoryError:
            result = dict(attention_chunk_blocks=chunk, out_of_memory=True)
        except RuntimeError as error:
            # Some torch_npu versions raise RuntimeError for allocator OOMs.
            if "out of memory" not in str(error).lower():
                raise
            result = dict(attention_chunk_blocks=chunk, out_of_memory=True)
        for tensor in tensors:
            tensor.grad = None
        if device.type != "cpu":
            getattr(torch, device.type).empty_cache()
        results.append(result)
    print(
        json.dumps(
            dict(
                torch_version=torch.__version__,
                device=str(device),
                configuration=vars(args),
                results=results,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
