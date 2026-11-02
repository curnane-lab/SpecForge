# coding=utf-8
"""Build a tiny pre-tokenized DFlash smoke dataset for the Qwen3.5-4B NPU gate.

Writes JSONL records of {"input_ids": [...], "loss_mask": [...]} so the run
does not depend on chat-template resolution. Each sample is a synthetic
prompt/completion pair: prompt tokens get loss_mask 0, completion tokens get
loss_mask 1 (long consecutive supervised runs satisfy the DFlash filter).
"""

import json
import os
import sys

from transformers import AutoTokenizer

MODEL_PATH = "/data1/f00538480/Qwen3.5-4B"
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smoke_dataset.jsonl")
NUM_SAMPLES = 8

PROMPTS = [
    "Explain why speculative decoding accelerates LLM inference.",
    "Summarize the role of a draft model in EAGLE-style training.",
    "Describe how Mooncake transfers hidden states between processes.",
    "What is the difference between greedy and typical acceptance?",
    "Write a short note on tensor parallelism for a 4B model.",
    "How does a hybrid GDN attention block differ from full attention?",
    "List three pitfalls when capturing auxiliary hidden states.",
    "Why must prefill capture keep every prompt token row?",
]

COMPLETIONS = [
    "Speculative decoding lets a cheap draft propose tokens that the target "
    "verifies in one parallel pass, so wall-clock latency drops while the "
    "output distribution stays identical to the target model.",
    "The draft model learns to imitate the target's next-token distribution "
    "from hidden states, trading a small quality budget for large speedups "
    "during autoregressive generation.",
    "Mooncake registers host memory segments and exposes object puts and "
    "gets over TCP or RDMA, letting a server publish features that a "
    "trainer reads without sharing a filesystem.",
    "Greedy acceptance requires an exact token match while typical "
    "acceptance tolerates plausible alternatives, raising acceptance length "
    "at a small risk to distribution fidelity.",
    "Tensor parallelism shards each matmul across devices and synchronizes "
    "activations with collectives, so a 4B model fits and runs faster when "
    "one device's memory or compute is the bottleneck.",
    "A GDN block replaces softmax attention with a gated delta rule over "
    "convolutional state, giving linear-time sequence mixing with a fixed "
    "state size instead of a growing KV cache.",
    "Capturing layers in the wrong order, silently dropping padded rows, "
    "and reusing graph buffers before the copy finishes all corrupt the "
    "training signal in ways that only appear much later.",
    "Every prompt token row feeds the draft's loss positions, so missing "
    "rows break the alignment between input ids and hidden states and the "
    "resulting gradients train on shifted features.",
]


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as out:
        for index in range(NUM_SAMPLES):
            prompt_ids = tokenizer.encode(PROMPTS[index % len(PROMPTS)])
            completion_ids = tokenizer.encode(COMPLETIONS[index % len(COMPLETIONS)])
            input_ids = prompt_ids + completion_ids
            loss_mask = [0] * len(prompt_ids) + [1] * len(completion_ids)
            out.write(json.dumps({"input_ids": input_ids, "loss_mask": loss_mask}) + "\n")
    print(f"wrote {NUM_SAMPLES} samples to {OUT_PATH}")


if __name__ == "__main__":
    sys.exit(main())
