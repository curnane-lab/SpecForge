"""N2 offline round-trip: plugin capture -> Mooncake -> offline .ckpt files.

Semantics equivalent to the upstream #1064 route (plugin -> Mooncake -> CPU
client -> file) while #1064 itself is unmerged: a real plugin capture server
publishes features for the fixed smoke dataset, a CPU client reads every
object back through Mooncake, and each sample is persisted in the offline
DFlash schema (one .ckpt per sample with input_ids / loss_mask /
hidden_states[/target_last_hidden_states]) for OfflineManifestReader.

Requires the same environment as verify_mooncake_e2e.py:
  - LD_LIBRARY_PATH=/data1/f00538480/cuda_stub_lib (mooncake wheel stubs)
  - mooncake_master -enable_http_metadata_server=true  (rpc 50051, http 8080)
  - PYTHONPATH=<sglang fork>/python:<specforge worktree>, acl paths intact

Usage: python offline_roundtrip.py [model_path] [dataset.jsonl] [out_dir]
"""

import json
import os
import sys

import torch

HIDDEN = 2560
DRAFT_CONFIG = "/data1/f00538480/specforge-npu-capture/configs/qwen3.5-4b-dflash.json"


def load_layer_ids():
    with open(DRAFT_CONFIG) as f:
        cfg = json.load(f)
    return [int(x) for x in cfg["dflash_config"]["target_layer_ids"]]


def main():
    argv = [a for a in sys.argv[1:] if a != "--batched"]
    model_path = argv[0] if len(argv) > 0 else "/data1/f00538480/Qwen3.5-4B"
    dataset_path = argv[1] if len(argv) > 1 else (
        "/data1/f00538480/specforge-npu-capture/scripts/npu/smoke/smoke_dataset.jsonl"
    )
    out_dir = argv[2] if len(argv) > 2 else (
        "/data1/f00538480/specforge-npu-capture/scripts/npu/smoke/offline_features"
    )
    layer_ids = load_layer_ids()
    os.makedirs(out_dir, exist_ok=True)
    os.environ["SPECFORGE_SPEC_CAPTURE"] = "1"

    samples = []
    with open(dataset_path) as f:
        for line in f:
            rec = json.loads(line)
            samples.append((rec["input_ids"], rec["loss_mask"]))
    print(f"loaded {len(samples)} samples from {dataset_path}", flush=True)

    import sglang as sgl

    engine = sgl.Engine(
        model_path=model_path,
        mem_fraction_static=0.5,
        page_size=int(os.environ.get("RT_PAGE_SIZE", "64")),
        disable_cuda_graph=True,
        mamba_ssm_dtype="bfloat16",
        log_level="info",
        enable_return_aux_hidden_states=True,
        speculative_algorithm="DFLASH",
        return_aux_hidden_state_layer_ids=layer_ids,
        return_hidden_states_mode="full",
        chunked_prefill_size=-1,
        skip_tokenizer_init=True,
        disable_radix_cache=True,
    )

    refs = []
    batched = "--batched" in sys.argv
    if batched:
        # One batch call: the scheduler prefills all prompts together, matching
        # how the online producer's requests land on the server. Feature bytes
        # then carry batched-prefill numerics rather than serial ones.
        prompts, masks = zip(*samples)
        specs = [
            {
                "store_id": "offline_rt",
                "sample_id": f"s{i}",
                "gen": 1,
                "replace": False,
                "features": {"aux": "hidden_states", "last_hidden": "target_last_hidden_states"},
                "passthrough": [
                    {"name": "input_ids", "data": list(ids), "shape": [1, len(ids)], "dtype": "int64"},
                    {"name": "loss_mask", "data": list(mask), "shape": [1, len(mask)], "dtype": "int64"},
                ],
            }
            for i, (ids, mask) in enumerate(zip(prompts, masks))
        ]
        outs = engine.generate(
            None,
            input_ids=[list(ids) for ids in prompts],
            sampling_params=[
                {
                    "max_new_tokens": 1,
                    "temperature": 0.0,
                    "custom_params": {"spec_capture": json.dumps(spec)},
                }
                for spec in specs
            ],
        )
        for i, out in enumerate(outs):
            result = (out.get("meta_info", {}).get("spec_capture") or [{}])[0]
            if "error" in result:
                print(f"FAIL: sample {i} publication error: {result['error'][:300]}")
                engine.shutdown()
                sys.exit(1)
            refs.append((i, len(samples[i][0])))
        print(f"captured {len(refs)} samples in one batched call", flush=True)
    else:
        for i, (input_ids, loss_mask) in enumerate(samples):
            spec = {
                "store_id": "offline_rt",
                "sample_id": f"s{i}",
                "gen": 1,
                "replace": False,
                "features": {"aux": "hidden_states", "last_hidden": "target_last_hidden_states"},
                "passthrough": [
                    {"name": "input_ids", "data": input_ids, "shape": [1, len(input_ids)], "dtype": "int64"},
                    {"name": "loss_mask", "data": loss_mask, "shape": [1, len(loss_mask)], "dtype": "int64"},
                ],
            }
            out = engine.generate(
                None,
                input_ids=input_ids,
                sampling_params={
                    "max_new_tokens": 1,
                    "temperature": 0.0,
                    "custom_params": {"spec_capture": json.dumps(spec)},
                },
            )
            result = (out.get("meta_info", {}).get("spec_capture") or [{}])[0]
            if "error" in result:
                print(f"FAIL: sample {i} publication error: {result['error'][:300]}")
                engine.shutdown()
                sys.exit(1)
            refs.append((i, len(input_ids)))
            print(f"captured sample {i}: L={len(input_ids)}", flush=True)

    # CPU client read-back; objects die with the engine, so read before shutdown.
    from mooncake.store import MooncakeDistributedStore

    store = MooncakeDistributedStore()
    rc = store.setup(
        local_hostname="localhost",
        metadata_server="http://localhost:8080/metadata",
        global_segment_size=0,
        local_buffer_size=0,
        protocol="tcp",
        rdma_devices="",
        master_server_addr="localhost:50051",
    )
    assert rc in (0, None), rc

    import time

    def read_back(key, shape, dtype):
        buf = torch.zeros(shape, dtype=dtype)
        ptr, nbytes = buf.data_ptr(), buf.numel() * buf.element_size()
        rc = store.register_buffer(ptr, nbytes)
        assert rc in (0, None), (key, rc)
        try:
            got = -800
            for attempt in range(4):  # TRANSFER_FAIL (-800) is retryable
                got = store.get_into(key, ptr, nbytes)
                if got == nbytes:
                    break
                time.sleep(1 + attempt)
            assert got == nbytes, (key, got, nbytes)
            return buf
        finally:
            # Registered regions must not overlap; free buffers get reused by
            # the caching allocator, so every registration is revoked here.
            store.unregister_buffer(ptr)

    failures = []
    for i, length in refs:
        prefix = f"offline_rt/s{i}/g1"
        input_ids = read_back(f"{prefix}/input_ids", (1, length), torch.int64)
        loss_mask = read_back(f"{prefix}/loss_mask", (1, length), torch.int64)
        hidden = read_back(f"{prefix}/hidden_states", (1, length, len(layer_ids) * HIDDEN), torch.bfloat16)
        last = read_back(f"{prefix}/target_last_hidden_states", (1, length, HIDDEN), torch.bfloat16)

        src_ids = samples[i][0]
        if input_ids[0].tolist() != src_ids:
            failures.append(f"sample {i}: input_ids mismatch")
            continue
        record = {
            "input_ids": input_ids[0].cpu(),
            "loss_mask": loss_mask[0].cpu(),
            "hidden_states": hidden.cpu(),
            "target_last_hidden_states": last.cpu(),
        }
        path = os.path.join(out_dir, f"sample_{i:05d}.ckpt")
        torch.save(record, path)

        # Round-trip integrity: reloading the file must reproduce the bytes.
        reloaded = torch.load(path, map_location="cpu", weights_only=False)
        for key, tensor in record.items():
            if not torch.equal(reloaded[key], tensor):
                failures.append(f"sample {i}: {key} changed across file round-trip")
        if not torch.isfinite(last.float()).all():
            failures.append(f"sample {i}: last_hidden not finite")
        print(f"wrote {path} (hidden {tuple(hidden.shape)})", flush=True)

    store.close()
    engine.shutdown()
    print("\n=== RESULT ===")
    if failures:
        for f in failures:
            print("FAIL:", f)
        sys.exit(1)
    print(f"VERIFY_OFFLINE_ROUNDTRIP_OK ({len(refs)} samples -> {out_dir})")


if __name__ == "__main__":
    main()
