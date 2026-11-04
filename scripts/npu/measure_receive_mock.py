"""Measure pageable and pinned receive copies with an in-memory Mooncake stub."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore
from tests.test_runtime.test_mooncake_store import _FakeMooncakeStore, _meta


def _payload():
    return {
        "hidden_state": torch.randn(1, 1024, 4096, dtype=torch.bfloat16),
        "input_ids": torch.arange(1024, dtype=torch.int64).unsqueeze(0),
        "loss_mask": torch.ones(1, 1024, dtype=torch.int64),
    }


def _timed_get(store, ref, device):
    started = time.perf_counter()
    output, handle = store.get(ref, device=device)
    torch.npu.synchronize(device)
    elapsed_ms = (time.perf_counter() - started) * 1000
    store.release(handle)
    return output, elapsed_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="npu:1")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    torch.npu.set_device(args.device)
    backend = _FakeMooncakeStore()
    producer = MooncakeFeatureStore(
        store=backend, store_id="run0", retain_on_release=True
    )
    expected = _payload()
    ref = producer.put(expected, sample_id="sample", metadata=_meta())
    consumers = {
        mode: MooncakeFeatureStore(
            store=backend,
            store_id="run0",
            receive_buffers=mode,
            retain_on_release=True,
        )
        for mode in ("pageable", "pinned")
    }
    timings = {mode: [] for mode in consumers}
    try:
        for mode, store in consumers.items():
            for _ in range(args.warmup):
                _timed_get(store, ref, args.device)
        for iteration in range(args.iterations):
            modes = ("pageable", "pinned")
            if iteration % 2:
                modes = modes[::-1]
            for mode in modes:
                _, elapsed_ms = _timed_get(consumers[mode], ref, args.device)
                timings[mode].append(elapsed_ms)
        for mode, store in consumers.items():
            output, _ = _timed_get(store, ref, args.device)
            for name, tensor in expected.items():
                assert torch.equal(output[name].cpu(), tensor), (mode, name)
    finally:
        for store in consumers.values():
            store.close()
        producer.close()

    result = {
        "commit": args.commit,
        "backend": "_FakeMooncakeStore (in-memory; no TCP/RDMA transport)",
        "device": args.device,
        "operation": "MooncakeFeatureStore.get(device) + torch.npu.synchronize(device)",
        "payload": {
            "hidden_state": {"shape": [1, 1024, 4096], "dtype": "bfloat16"},
            "input_ids": {"shape": [1, 1024], "dtype": "int64"},
            "loss_mask": {"shape": [1, 1024], "dtype": "int64"},
            "bytes": sum(t.numel() * t.element_size() for t in expected.values()),
            "mib": sum(t.numel() * t.element_size() for t in expected.values()) / 2**20,
        },
        "warmup_per_mode": args.warmup,
        "iterations_per_mode": args.iterations,
        "ordering": "alternating pageable/pinned order per iteration",
        "results": {
            mode: {
                "median_ms": statistics.median(samples),
                "mean_ms": statistics.mean(samples),
                "min_ms": min(samples),
                "max_ms": max(samples),
                "samples_ms": samples,
            }
            for mode, samples in timings.items()
        },
    }
    result["speedup_pageable_over_pinned"] = (
        result["results"]["pageable"]["median_ms"]
        / result["results"]["pinned"]["median_ms"]
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "results"}, indent=2))
    for mode, samples in result["results"].items():
        print(mode, {k: v for k, v in samples.items() if k != "samples_ms"})
    print("speedup:", result["speedup_pageable_over_pinned"])
    print("MEASURE_RECEIVE_MOCK_OK")


if __name__ == "__main__":
    main()
