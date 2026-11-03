"""N3 hardware verification: pinned receive + device delivery on Ascend NPU.

Through the real Mooncake TCP transport (not the fake backend):
  - consumer_device() reports the NPU for a pinned consumer;
  - get(ref, device="npu:0") delivers float features on the NPU, integer
    features on the host, all values exact;
  - alternating A/B reads with immediate slot reuse never overwrite an earlier
    output (per-copy completion is enforced before the slot returns);
  - release() then get() raises KeyError (no use-after-free);
  - timing is printed against the pageable baseline (~16.5 ms per 40 MiB
    sample: 11.4 ms get + 5.1 ms trainer-thread H2D).

Requires mooncake_master on localhost:50051/8080 and the CUDA stub libs.
"""

import os
import time

import torch

from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore

L, AUX, HID = 2048, 7680, 2560


def make_tensors(seed, value):
    g = torch.Generator().manual_seed(seed)
    return {
        "hidden_states": torch.full((1, L, AUX), value, dtype=torch.bfloat16)
        + torch.randn(1, L, AUX, generator=g, dtype=torch.float32).bfloat16() * 0.01,
        "target": torch.randn(1, L, HID, generator=g, dtype=torch.float32).bfloat16(),
        "input_ids": torch.randint(0, 150000, (1, L), generator=g, dtype=torch.int64),
        "loss_mask": torch.ones(1, L, dtype=torch.int64),
    }


def main():
    os.environ["LOCAL_RANK"] = "0"
    torch.npu.set_device(0)
    kw = {
        "local_hostname": "localhost",
        "metadata_server": "http://localhost:8080/metadata",
        "master_server_addr": "localhost:50051",
    }
    producer = MooncakeFeatureStore(store_id="n3hw", setup_kwargs=kw)
    consumer = MooncakeFeatureStore(
        store=producer._store, store_id="n3hw", receive_buffers="pinned"
    )
    device = consumer.consumer_device()
    assert device == torch.device("npu", 0), device
    print("consumer_device:", device)

    samples = {"a": make_tensors(0, 7.0), "b": make_tensors(1, 11.0)}
    refs = {k: producer.put(v, sample_id=k, metadata={}) for k, v in samples.items()}
    got = {}
    handles = []
    for name in ("a", "b", "a", "b"):
        out, handle = consumer.get(refs[name], device="npu:0")
        handles.append(handle)
        expected = samples[name]
        assert out["hidden_states"].device.type == "npu", out["hidden_states"].device
        assert out["input_ids"].device.type == "cpu"
        for key, tensor in expected.items():
            assert torch.equal(out[key].cpu(), tensor), (name, key)
        got[name] = out  # keep both alive; a later read must not clobber them
        print(f"read {name}: exact (npu delivery, host integer features)")
    for name in ("a", "b"):
        for key, tensor in samples[name].items():
            assert torch.equal(got[name][key].cpu(), tensor), ("late reuse", name, key)
    print("A/B slot-reuse: earlier outputs intact after alternating reads")
    print("pool health:", consumer._receive_pool.health())

    t0 = time.perf_counter()
    for name in ("a", "b"):
        _, h = consumer.get(refs[name], device="npu:0")
        handles.append(h)
    dt = (time.perf_counter() - t0) / 2
    print(f"get(npu) pinned total: {dt * 1e3:.1f} ms/sample (pageable baseline ~16.5 ms)")

    _, handle = consumer.get(refs["a"], device="npu:0")
    handles.append(handle)
    for h in handles:
        consumer.release(h)
    try:
        consumer.get(refs["a"], device="npu:0")
        raise AssertionError("get after release did not raise")
    except KeyError:
        print("release -> get raises KeyError (no use-after-free)")

    consumer.close()
    producer.close()
    print("VERIFY_NPU_PINNED_RECEIVE_OK")


if __name__ == "__main__":
    main()
