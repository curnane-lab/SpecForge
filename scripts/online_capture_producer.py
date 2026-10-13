"""Scratch-online capture producer: generate hidden states on demand.

Companion to ``data.scratch_online: true`` training. Runs the same local
SGLang capture as ``prepare_hidden_states.py``, but instead of writing the
whole dataset up front it keeps only a sliding window of unconsumed feature
files in the output directory: the trainer (via ``ScratchFileFeatureStore``)
waits for each file and deletes it after reading, so disk usage stays bounded
by ``--scratch-window`` samples instead of growing with the dataset.

The protocol is file-based (in the spirit of the H-Spec authors'
speculators/vLLM ``--on-missing generate --on-generate delete`` flow):
files are published atomically (tmp + rename), consumed in index order, and
regenerated from index 0 when ``--scratch-loop`` is set (multi-epoch runs).

Usage (single NPU/GPU for capture; the trainer runs separately):

    python scripts/online_capture_producer.py \
        --target-model-path Qwen/Qwen3-4B \
        --data-path sharegpt.jsonl \
        --output-path ./scratch \
        --strategy hspec \
        --draft-model-config configs/qwen3-4b-hspec-reference.json \
        --chat-template qwen --max-length 3072 --batch-size 4 \
        --scratch-window 64 --scratch-loop
"""

import argparse
import hashlib
import os
import sys
import time

from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoConfig

# The producer's own flags are stripped from argv before delegating to
# prepare_hidden_states.parse_args() so the two CLIs stay in lockstep.
from scripts.prepare_hidden_states import (
    HiddenStatesGenerator,
    _generate_shared_vocab_mapping,
    _resolve_draft_vocab_size,
    build_target_model,
    parse_args,
    resolve_offline_capture_plan,
)
from specforge.data.preprocessing import build_eagle3_dataset
from specforge.data.utils import DataCollatorWithPadding
from specforge.distributed import destroy_distributed, init_distributed
from specforge.utils import (
    is_xpu,
    load_tokenizer,
    print_with_rank,
    rank_0_priority,
    safe_conversations_generator,
)


def _parse_producer_args(argv):
    parser = argparse.ArgumentParser(
        description="Scratch-online capture producer (windowed prepare_hidden_states)"
    )
    parser.add_argument(
        "--scratch-window",
        type=int,
        default=32,
        help=(
            "Max unconsumed feature files kept on disk. Must exceed the "
            "trainer's dataloader prefetch depth (dataloader_num_workers + "
            "batch headroom) to keep the pipeline full."
        ),
    )
    parser.add_argument(
        "--scratch-loop",
        action="store_true",
        help="Wrap to index 0 after the last sample (multi-epoch training).",
    )
    parser.add_argument(
        "--scratch-poll-interval",
        type=float,
        default=2.0,
        help="Seconds between window-capacity checks (default: 2).",
    )
    own, remaining = parser.parse_known_args(argv)
    return own, remaining


def _count_unconsumed(output_path: str) -> int:
    count = 0
    for _root, _dirs, names in os.walk(output_path):
        count += sum(1 for n in names if n.endswith((".ckpt", ".ckpt.gz")))
    return count


class _AtomicHiddenStatesGenerator(HiddenStatesGenerator):
    """Publishes feature files atomically so consumers never see partial writes."""

    def _save_tensor_sync(self, record, output_file: str) -> None:
        tmp = output_file + ".tmp"
        super()._save_tensor_sync(record, tmp)
        if os.path.exists(tmp):
            os.replace(tmp, output_file)
        else:
            # The base implementation silently skips NaN records. Publish a
            # poison marker instead, so the consumer fails fast with an
            # actionable error rather than waiting for a file that never comes.
            with open(output_file + ".nan_skip", "w"):
                pass


def main():
    own_args, remaining = _parse_producer_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + remaining
    args = parse_args()
    if is_xpu():
        args.sglang_attention_backend = "triton"
    if args.num_io_threads is None:
        args.num_io_threads = max(1, os.cpu_count() or 1)
    if args.output_path is None:
        raise ValueError("--output-path (the scratch dir) is required")

    target_model_config = AutoConfig.from_pretrained(
        args.target_model_path,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
    )
    capture_plan = resolve_offline_capture_plan(args, target_model_config)
    draft_vocab_size = _resolve_draft_vocab_size(capture_plan.draft_config)

    # Single-process capture: window accounting assumes one producer stream.
    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
    print_with_rank(
        f"Resolved {capture_plan.strategy} scratch-online capture, "
        f"method {capture_plan.capture_method!r}, layers: "
        f"{list(capture_plan.capture_layers)}, window: {own_args.scratch_window}"
    )

    target_model = build_target_model(
        args,
        target_model_config,
        capture_layers=list(capture_plan.capture_layers),
        capture_method=capture_plan.capture_method,
        kv_layer_ids=capture_plan.kv_layer_ids,
    )
    target_text_config = getattr(
        target_model_config, "text_config", target_model_config
    )
    target_vocab_size = int(target_text_config.vocab_size)
    if draft_vocab_size > target_vocab_size:
        raise ValueError(
            f"draft_vocab_size={draft_vocab_size} exceeds target "
            f"vocab_size={target_vocab_size}"
        )

    assert os.path.exists(
        args.data_path
    ), f"Dataset path {args.data_path} does not exist"
    with rank_0_priority():
        print_with_rank("Loading/building dataset cache...")
        dataset = Dataset.from_generator(
            generator=safe_conversations_generator,
            gen_kwargs={"file_path": args.data_path},
            num_proc=min(args.build_dataset_num_proc, 32),
        )
    if args.num_samples is not None and capture_plan.loss_mask_filter is None:
        dataset = dataset.select(range(args.num_samples))
    tokenizer = load_tokenizer(
        args.target_model_path, trust_remote_code=args.trust_remote_code
    )
    cache_params_string = (
        f"{args.data_path}-{args.max_length}-{args.chat_template}"
        f"-{args.target_model_path}-{args.num_samples}-{args.is_preformatted}"
    )
    cache_key = hashlib.md5(cache_params_string.encode()).hexdigest()
    with rank_0_priority():
        print_with_rank("Main process is building the dataset cache...")
        eagle3_dataset = build_eagle3_dataset(
            dataset=dataset,
            tokenizer=tokenizer,
            chat_template=args.chat_template,
            max_length=args.max_length,
            cache_dir=os.path.join(args.cache_dir, "processed_dataset"),
            cache_key=cache_key,
            is_preformatted=args.is_preformatted,
            num_proc=args.build_dataset_num_proc,
            loss_mask_filter=capture_plan.loss_mask_filter,
        )
    if capture_plan.loss_mask_filter is not None and args.num_samples is not None:
        eagle3_dataset = eagle3_dataset.select(
            range(min(args.num_samples, len(eagle3_dataset)))
        )
    if not len(eagle3_dataset):
        raise ValueError(
            f"no samples satisfy {capture_plan.strategy} training eligibility"
        )
    total = len(eagle3_dataset)
    print(f"[producer] dataset prepared: {total} samples", flush=True)

    vocab_mapping_path = _generate_shared_vocab_mapping(
        eagle3_dataset,
        output_path=args.output_path,
        target_vocab_size=target_vocab_size,
        draft_vocab_size=draft_vocab_size,
    )
    print_with_rank(f"Vocabulary mapping ready at {vocab_mapping_path}")

    next_idx = 0
    try:
        with _AtomicHiddenStatesGenerator(
            target_model,
            capture_layout=capture_plan.layout,
            capture_method=capture_plan.capture_method,
            num_io_threads=args.num_io_threads,
            io_queue_size=args.io_queue_size,
            file_group_size=args.file_group_size,
            compress=args.compress,
            compression_level=args.compression_level,
        ) as generator:
            while True:
                if next_idx >= total:
                    if not own_args.scratch_loop:
                        break
                    next_idx = 0
                unconsumed = _count_unconsumed(args.output_path)
                slots = own_args.scratch_window - unconsumed
                if slots <= 0:
                    time.sleep(own_args.scratch_poll_interval)
                    continue
                hi = min(next_idx + slots, total)
                chunk = eagle3_dataset.select(range(next_idx, hi))
                # Plain DataLoader, not prepare_dp_dataloaders: the producer is
                # a single stream (dp=1), and the capture loader's drop_last=True
                # would silently drop every chunk's partial tail batch, stranding
                # those indices forever (the consumer then times out waiting).
                data_loader = DataLoader(
                    chunk,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    collate_fn=DataCollatorWithPadding(),
                    drop_last=False,
                )
                print(
                    f"[producer] producing samples [{next_idx}, {hi}) "
                    f"({unconsumed} unconsumed in window)",
                    flush=True,
                )
                generator.generate(
                    data_loader,
                    output_path=args.output_path,
                    start_idx=next_idx,
                    samples_per_dp=len(chunk),
                )
                next_idx = hi
    finally:
        print_with_rank("Scratch-online producer finished.")
        destroy_distributed()


if __name__ == "__main__":
    main()
