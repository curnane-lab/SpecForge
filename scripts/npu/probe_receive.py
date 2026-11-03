"""N3 decision probe: measure the NPU receive path before optimizing it.

Puts realistic DFlash samples into Mooncake through SpecForge's own
MooncakeFeatureStore, then times the current pageable receive path and the
H2D copy a trainer pays on its critical path. Also probes whether this
torch_npu supports pinned host allocation and async H2D on an NPU stream.

Run: mooncake_master on localhost:50051/8080, LD_LIBRARY_PATH=cuda_stub_lib,
ASCEND_RT_VISIBLE_DEVICES=1, PYTHONPATH=<specforge>.
"""

import os
import time

import torch

from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore

L = 2048
AUX = 7680
HID = 2560
N_SAMPLES = 8


def make_tensors(seed):
    g = torch.Generator().manual_seed(seed)
    return {
        "hidden_states": torch.randn(1, L, AUX, generator=g, dtype=torch.float32).bfloat16(),
        "target": torch.randn(1, L, HID, generator=g, dtype=torch.float32).bfloat16(),
        "input_ids": torch.randint(0, 150000, (1, L), generator=g, dtype=torch.int64),
        "loss_mask": torch.ones(1, L, dtype=torch.int64),
    }


def nbytes(tensors):
    return sum(t.numel() * t.element_size() for t in tensors.values())


def main():
    print("torch_npu present:", hasattr(torch, "npu"), "available:", torch.npu.is_available())
    torch.npu.set_device(0)
    store = MooncakeFeatureStore(
        store_id="n3probe",
        retain_on_release=True,
        setup_kwargs={
            "local_hostname": "localhost",
            "metadata_server": "http://localhost:8080/metadata",
            "master_server_addr": "localhost:50051",
        },
    )
    refs = []
    tensors = {i: make_tensors(i) for i in range(N_SAMPLES)}
    per_sample = nbytes(tensors[0]) / 2**20
    print(f"per-sample payload: {per_sample:.1f} MiB x {N_SAMPLES}")

    t0 = time.perf_counter()
    for i in range(N_SAMPLES):
        refs.append(store.put(tensors[i], sample_id=f"s{i}", metadata={}))
    print(f"put: {(time.perf_counter() - t0) / N_SAMPLES * 1e3:.1f} ms/sample")

    # Warm the transport once.
    out, handle = store.get(refs[0], device="cpu")
    store.release(handle)

    t0 = time.perf_counter()
    for ref in refs:
        out, handle = store.get(ref, device="cpu")
        store.release(handle)
    get_cpu = (time.perf_counter() - t0) / N_SAMPLES
    print(f"get(cpu), pageable receive: {get_cpu * 1e3:.1f} ms/sample")

    out, handle = store.get(refs[0], device="cpu")
    h2d_ms = 0.0
    for _ in range(3):
        t0 = time.perf_counter()
        dev = {k: v.to("npu:0") for k, v in out.items()}
        torch.npu.synchronize()
        h2d_ms += (time.perf_counter() - t0) * 1e3 / 3
    print(f"H2D pageable->npu (blocking): {h2d_ms:.1f} ms/sample")
    print(f"=> current get(npu) total: {(get_cpu * 1e3 + h2d_ms):.1f} ms/sample")

    # Pinned-host support on this torch_npu build.
    try:
        pinned = torch.empty(64 << 20, dtype=torch.uint8, pin_memory=True)
        print("pinned alloc: OK", "is_pinned =", pinned.is_pinned())
    except Exception as exc:
        print("pinned alloc: FAILED:", exc)
        pinned = None
    if pinned is not None:
        src = out["hidden_states"]
        for label, buf in (("pageable", src), ("pinned", src.pin_memory())):
            dst = torch.empty_like(src, device="npu:0")
            torch.npu.synchronize()
            t0 = time.perf_counter()
            for _ in range(5):
                dst.copy_(buf, non_blocking=False)
            torch.npu.synchronize()
            dt = (time.perf_counter() - t0) / 5
            mb = src.numel() * src.element_size() / 2**20
            print(f"H2D {label}: {dt * 1e3:.2f} ms for {mb:.0f} MiB -> {mb / 2**10 / dt:.2f} GiB/s")
        # Async H2D on a side stream + event.
        try:
            stream = torch.npu.Stream()
            dst = torch.empty_like(src, device="npu:0")
            with torch.npu.stream(stream):
                dst.copy_(src.pin_memory(), non_blocking=True)
                ev = torch.npu.Event()
                ev.record()
            ev.synchronize()
            print("async H2D on npu stream + event: OK; exact =", torch.equal(dst, src.npu()))
        except Exception as exc:
            print("async H2D on npu stream: FAILED:", exc)
    store.release(handle)
    store.close()
    print("PROBE_OK")


if __name__ == "__main__":
    main()
