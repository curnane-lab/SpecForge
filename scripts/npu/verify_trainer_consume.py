"""N2 gate (trainer side): consume a published sample through SpecForge's own
MooncakeFeatureStore on Ascend NPU.

Complements verify_mooncake_e2e.py (raw store read-back) by exercising the
real consumer path a trainer uses: SampleRef -> MooncakeFeatureStore.get() ->
CPU receive -> optional H2D move -> release().

Requires the same Mooncake setup as verify_mooncake_e2e.py (master running,
LD_LIBRARY_PATH stubs on this CUDA-less host).

Checks:
  1. get(ref, "cpu") returns every feature; aux matches the engine's own
     response rows bit-for-bit.
  2. get(ref, "npu:0") is the baseline receive-then-H2D path; values match.
  3. release() frees the objects; a later get() raises (no use-after-free).
"""

import json
import os
import sys

import torch

LAYER_IDS = [2, 16, 29]
HIDDEN = 2560
INPUT_IDS = [151644, 872, 198, 6023, 374, 11914, 30, 151645, 198]
STORE_ID, SAMPLE_ID, GEN = "train", "s0", 1


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "/data1/f00538480/Qwen3.5-4B"
    os.environ["SPECFORGE_SPEC_CAPTURE"] = "1"

    import sglang as sgl

    engine = sgl.Engine(
        model_path=model_path,
        mem_fraction_static=0.5,
        page_size=64,
        disable_cuda_graph=True,
        mamba_ssm_dtype="bfloat16",
        log_level="warning",
        enable_return_aux_hidden_states=True,
        speculative_algorithm="DFLASH",
        return_aux_hidden_state_layer_ids=LAYER_IDS,
        return_hidden_states_mode="full",
        chunked_prefill_size=-1,
        skip_tokenizer_init=True,
        disable_radix_cache=True,
    )
    spec = {
        "store_id": STORE_ID,
        "sample_id": SAMPLE_ID,
        "gen": GEN,
        "replace": False,
        "features": {"aux": "hidden_states", "last_hidden": "target"},
        "passthrough": [
            {"name": "input_ids", "data": INPUT_IDS, "shape": [1, len(INPUT_IDS)], "dtype": "int64"}
        ],
    }
    out = engine.generate(
        None,
        input_ids=INPUT_IDS,
        sampling_params={
            "max_new_tokens": 1,
            "temperature": 0.0,
            "custom_params": {"spec_capture": json.dumps(spec)},
        },
    )
    result = out["meta_info"]["spec_capture"][0]
    assert "error" not in result, result["error"][:300]
    features = result["features"]
    print("published:", json.dumps(features), flush=True)

    ref_rows = engine.generate(
        None,
        input_ids=INPUT_IDS,
        sampling_params={"max_new_tokens": 1, "temperature": 0.0},
        return_hidden_states=True,
    )
    n_prompt = ref_rows["meta_info"]["prompt_tokens"]
    rows = [r for block in ref_rows["meta_info"]["hidden_states"] for r in block]
    ref_aux = torch.tensor(rows[:n_prompt], dtype=torch.float32)

    # ---- trainer-side consume through the real data plane -----------------
    from specforge.runtime.contracts import FeatureSpec, SampleRef
    from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore

    sample_ref = SampleRef(
        sample_id=SAMPLE_ID,
        run_id="n2-smoke",
        source_task_id=None,
        feature_store_uri="mooncake://localhost",
        feature_keys={name: name for name in features},
        feature_specs={
            name: FeatureSpec(name=name, shape=tuple(meta["shape"]), dtype=meta["dtype"])
            for name, meta in features.items()
        },
        strategy="dflash",
        metadata={"generation": GEN},
    )
    store = MooncakeFeatureStore(
        store_id=STORE_ID,
        setup_kwargs={
            "local_hostname": "localhost",
            "metadata_server": "http://localhost:8080/metadata",
            "master_server_addr": "localhost:50051",
            "protocol": "tcp",
            "rdma_devices": "",
            "global_segment_size": 0,
            "local_buffer_size": 0,
        },
    )
    failures = []

    print("=== get(ref, cpu) ===", flush=True)
    tensors, handle = store.get(sample_ref, device="cpu")
    aux = tensors["hidden_states"][0].float()
    same = aux.shape == ref_aux.shape and torch.equal(aux, ref_aux)
    print(f"aux rows match engine: {same}")
    if not same:
        failures.append("trainer-side aux payload != engine rows")
    ids = tensors["input_ids"][0].tolist()
    if ids != INPUT_IDS:
        failures.append(f"trainer-side input_ids mismatch: {ids}")
    if not torch.isfinite(tensors["target"].float()).all():
        failures.append("trainer-side target has non-finite values")

    print("=== get(ref, 'npu:0') baseline H2D ===", flush=True)
    try:
        npu_tensors, handle2 = store.get(sample_ref, device="npu:0")
        dev_ok = all(str(t.device).startswith("npu") for t in npu_tensors.values())
        moved = npu_tensors["hidden_states"][0].float().cpu()
        same_npu = torch.equal(moved, ref_aux)
        print(f"devices npu: {dev_ok}; aux after H2D matches: {same_npu}")
        if not (dev_ok and same_npu):
            failures.append("npu:0 receive path mismatch")
    except Exception as e:  # baseline path may legitimately lack NPU support
        failures.append(f"get(device='npu:0') failed: {type(e).__name__}: {e}")

    print("=== release then get must fail ===", flush=True)
    store.release(handle)
    store.release(handle2)
    try:
        store.get(sample_ref, device="cpu")
        failures.append("get() after release did not raise (use-after-free window)")
    except (KeyError, Exception) as e:
        print(f"post-release get raised as expected: {type(e).__name__}")

    store.close()
    engine.shutdown()
    print("\n=== RESULT ===")
    if failures:
        for f in failures:
            print("FAIL:", f)
        sys.exit(1)
    print("VERIFY_TRAINER_OK")


if __name__ == "__main__":
    main()
