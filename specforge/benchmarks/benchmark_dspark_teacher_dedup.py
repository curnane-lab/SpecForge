"""DSpark teacher projection + CE/L1/confidence forward/backward microbenchmark.

Three arms distinguish teacher row deduplication from the indexed L1 backend:
original dense, dense teacher with identity row indices, deduplicated teacher.
Includes gathers, unique/inverse construction, projection, loss and checkpoint
replay. Excludes the draft backbone, Markov head, optimizer and teacher forward.
Synthetic anchors and features cannot establish full-training throughput.
"""

import argparse
import gc
import json
import math
import statistics
import time
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from specforge.core.chunking import checkpointed_chunk_reduce
from specforge.core.fused_loss import (
    ce_l1_loss,
    fused_ce_l1_loss,
    supports_fused_ce_l1_loss,
)
from specforge.core.indexed_loss import indexed_ce_l1_loss, supports_indexed_loss
from specforge.core.teacher_projection import (
    IndexedTeacherLogits,
    project_teacher_positions,
    teacher_position_keys,
)


def chunk_terms(hidden, keys, labels, mask, *, arm, source, head, confidence, fused):
    logits = head(hidden)
    if arm == "dedup_indexed":
        teacher = project_teacher_positions(source, keys, head)
    else:
        with torch.no_grad():
            aligned = source[keys // source.shape[1], keys % source.shape[1]]
            teacher = head(aligned)
        if arm == "dense_indexed":
            teacher = IndexedTeacherLogits(
                teacher.reshape(-1, teacher.shape[-1]),
                torch.arange(keys.numel(), device=keys.device).reshape_as(keys),
            )
    if arm == "dense":
        compute = (
            fused_ce_l1_loss
            if fused and supports_fused_ce_l1_loss(logits)
            else ce_l1_loss
        )
        terms = compute(logits, labels, teacher_logits=teacher)
    else:
        terms = indexed_ce_l1_loss(logits, labels, teacher_logits=teacher, fused=fused)
    acceptance = (1 - terms.teacher_l1.detach() / 2).clamp(0, 1)
    confidence_loss = F.binary_cross_entropy_with_logits(
        confidence(hidden).squeeze(-1).float(), acceptance, reduction="none"
    )
    return (
        (
            mask
            * (0.1 * terms.cross_entropy + 0.9 * terms.teacher_l1 + confidence_loss)
        ).sum(),
    )


def run(args):
    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)
    accelerator = getattr(torch, device.type) if device.type != "cpu" else None
    if accelerator is not None:
        accelerator.set_device(device)
    synchronize = accelerator.synchronize if accelerator is not None else lambda: None
    source = torch.randn(
        args.batch, args.sequence, args.hidden, device=device, dtype=dtype
    )
    hidden = torch.randn(
        args.batch,
        args.anchors,
        args.block_size,
        args.hidden,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    head = nn.Linear(
        args.hidden, args.vocab, bias=False, device=device, dtype=dtype
    ).requires_grad_(False)
    confidence = nn.Linear(args.hidden, 1, device=device, dtype=dtype)
    # Match DSpark's sorted sampling without replacement and tail clamping.
    start = (
        int((args.sequence - 1) * (1 - args.suffix_fraction))
        if args.pattern == "suffix"
        else 0
    )
    available = args.sequence - 1 - start
    if args.anchors > available:
        raise ValueError("anchors exceed the supervised sampling region")
    anchors = torch.rand(args.batch, available, device=device).argsort(-1)[
        :, : args.anchors
    ]
    anchors = (anchors + start).sort(-1).values
    label_positions = (
        anchors[..., None] + 1 + torch.arange(args.block_size, device=device)
    )
    keys = teacher_position_keys(
        source, label_positions.clamp(max=args.sequence - 1) - 1
    )
    mask = (label_positions < args.sequence).float()
    labels = torch.randint(args.vocab, mask.shape, device=device)
    chunk = args.chunk or args.anchors
    unique_rows = [
        keys[:, i : i + chunk].unique().numel() for i in range(0, args.anchors, chunk)
    ]
    proposal_rows = keys.numel()
    backend = (
        "triton" if args.fused and supports_indexed_loss(hidden) else "torch_tiled"
    )
    results = {
        "settings": vars(args),
        "torch": torch.__version__,
        "indexed_l1_backend": backend,
        "proposal_rows_per_forward": proposal_rows,
        "unique_teacher_rows_per_forward": sum(unique_rows),
        "unique_rows_per_chunk": unique_rows,
        "projected_row_reduction": 1 - sum(unique_rows) / proposal_rows,
        "scope": __doc__,
        "arms": {},
    }
    reference_loss = reference_gradient = None
    for arm in ("dense", "dense_indexed", "dedup_indexed"):
        project_rows = []

        def count_teacher_rows(_module, inputs):
            if not torch.is_grad_enabled():
                project_rows.append(inputs[0].numel() // args.hidden)

        hook = head.register_forward_pre_hook(count_teacher_rows)

        def step():
            hidden.grad = None
            confidence.zero_grad(set_to_none=True)
            loss = (
                checkpointed_chunk_reduce(
                    partial(
                        chunk_terms,
                        arm=arm,
                        source=source,
                        head=head,
                        confidence=confidence,
                        fused=args.fused,
                    ),
                    hidden,
                    keys,
                    labels,
                    mask,
                    chunk_size=args.chunk,
                    dim=1,
                    checkpoint=args.checkpoint,
                )[0]
                / mask.sum()
            )
            loss.backward()
            return loss.detach()

        loss = step()
        synchronize()
        projected_rows = sum(project_rows)
        hook.remove()  # Python instrumentation is excluded from timing.
        gradient = hidden.grad.detach().float().cpu()
        if reference_gradient is None:
            reference_loss, reference_gradient = loss.item(), gradient
        difference = gradient - reference_gradient
        error = {
            "loss": loss.item(),
            "loss_abs_error": abs(loss.item() - reference_loss),
            "gradient_relative_l2": (
                difference.norm() / reference_gradient.norm().clamp_min(1e-30)
            ).item(),
            "gradient_max_abs": difference.abs().max().item(),
        }
        if not torch.isfinite(gradient).all() or not math.isfinite(error["loss"]):
            raise RuntimeError(f"{arm}: non-finite loss or gradient")
        if error["loss_abs_error"] > 5e-3 or error["gradient_relative_l2"] > 3e-2:
            raise RuntimeError(f"{arm}: correctness check failed: {error}")
        for _ in range(args.warmup):
            step()
        hidden.grad = None
        confidence.zero_grad(set_to_none=True)
        gc.collect()
        if accelerator is not None:
            accelerator.empty_cache()
            synchronize()
            accelerator.reset_peak_memory_stats()
        timings = []
        for _ in range(args.repeats):
            synchronize()
            before = time.perf_counter()
            loss = step()
            synchronize()
            timings.append((time.perf_counter() - before) * 1000)
        results["arms"][arm] = {
            "median_ms": statistics.median(timings),
            "step_ms": timings,
            "teacher_rows_including_checkpoint_replay": projected_rows,
            "peak_allocated_gib": (
                accelerator.max_memory_allocated() / 2**30 if accelerator else None
            ),
            "peak_reserved_gib": (
                accelerator.max_memory_reserved() / 2**30 if accelerator else None
            ),
            **error,
        }
    results["dedup_speedup_vs_dense"] = (
        results["arms"]["dense"]["median_ms"]
        / results["arms"]["dedup_indexed"]["median_ms"]
    )
    results["dedup_speedup_vs_dense_indexed"] = (
        results["arms"]["dense_indexed"]["median_ms"]
        / results["arms"]["dedup_indexed"]["median_ms"]
    )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype", choices=["float32", "bfloat16", "float16"], default="float32"
    )
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--sequence", type=int, default=128)
    parser.add_argument("--anchors", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=7)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--vocab", type=int, default=1024)
    parser.add_argument("--chunk", type=int, default=16)
    parser.add_argument("--pattern", choices=["uniform", "suffix"], default="uniform")
    parser.add_argument("--suffix-fraction", type=float, default=0.25)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--fused", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--checkpoint", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--output", help="Optional JSON result path")
    args = parser.parse_args()
    if (
        min(
            args.batch,
            args.anchors,
            args.block_size,
            args.hidden,
            args.vocab,
            args.repeats,
        )
        < 1
        or args.sequence < 2
    ):
        parser.error("shapes/repeats must be positive; sequence must be >= 2")
    if min(args.chunk, args.warmup) < 0 or not 0 < args.suffix_fraction <= 1:
        parser.error("chunk/warmup must be >= 0; suffix-fraction must be in (0, 1]")
    result = run(args)
    rendered = json.dumps(result, indent=2)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
