"""N2 entry: capture -> Mooncake -> client read-back on Ascend NPU.

Host publication over TCP. Requires:
  - mooncake-transfer-engine wheel installed (0.3.13.post1 here), with the
    CUDA-linked native libs resolved through the local stubs:
    LD_LIBRARY_PATH=/data1/f00538480/cuda_stub_lib
  - mooncake_master running locally:
    mooncake_master -enable_http_metadata_server=true   (rpc 50051, http 8080)

Checks:
  1. A spec_capture request is deferred until published and returns feature
     refs (no error) on the NPU stack.
  2. An independent client store reads each object back: the passthrough
     input_ids are byte-exact; the aux payload equals, bit for bit, the aux
     rows the same engine returns for a follow-up return_hidden_states
     request on the same prompt (radix disabled, so it re-prefills);
     last_hidden has the right shape/dtype and is finite.
"""

import json
import os
import sys
import time

import torch

LAYER_IDS = [2, 16, 29]
HIDDEN = 2560
INPUT_IDS = [151644, 872, 198, 6023, 374, 11914, 30, 151645, 198]
STORE_ID, SAMPLE_ID, GEN = "e2e", "s0", 1


def rel_err(a, b):
    return (a - b).norm().item() / b.norm().clamp_min(1e-6).item()


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
        log_level="info",
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
    failures = []

    print("=== capture request (expect publication, not error) ===", flush=True)
    out = engine.generate(
        None,
        input_ids=INPUT_IDS,
        sampling_params={
            "max_new_tokens": 1,
            "temperature": 0.0,
            "custom_params": {"spec_capture": json.dumps(spec)},
        },
    )
    meta = out.get("meta_info", {})
    sc = meta.get("spec_capture")
    assert sc, "spec_capture missing from meta_info"
    result = sc[0]
    print("spec_capture result:", json.dumps(result)[:400])
    if "error" in result:
        print("FAIL: publication error:", result["error"][:300])
        engine.shutdown()
        sys.exit(1)
    features = result.get("features") or {}
    assert features, f"no features in result: {result}"

    print("=== follow-up plain request for reference rows ===", flush=True)
    ref = engine.generate(
        None,
        input_ids=INPUT_IDS,
        sampling_params={"max_new_tokens": 1, "temperature": 0.0},
        return_hidden_states=True,
    )
    n_prompt = ref["meta_info"]["prompt_tokens"]
    rows = [r for block in ref["meta_info"]["hidden_states"] for r in block]
    ref_aux = torch.tensor(rows[:n_prompt], dtype=torch.float32)  # [n, 3H]

    # Read back while the engine is still up: the mounted segment lives in the
    # scheduler process, so objects die with it.
    print("=== client read-back ===", flush=True)
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

    def read_back(name, shape, dtype):
        key = f"{STORE_ID}/{SAMPLE_ID}/g{GEN}/{name}"
        buf = torch.zeros(shape, dtype=dtype)
        ptr, nbytes = buf.data_ptr(), buf.numel() * buf.element_size()
        rc = store.register_buffer(ptr, nbytes)
        assert rc in (0, None), (key, rc)
        got = -800
        for attempt in range(4):  # TRANSFER_FAIL (-800) is retryable
            got = store.get_into(key, ptr, nbytes)
            if got == nbytes:
                break
            time.sleep(1 + attempt)
        assert got == nbytes, (key, got, nbytes)
        return buf

    ids = read_back("input_ids", (1, len(INPUT_IDS)), torch.int64)
    if ids[0].tolist() != INPUT_IDS:
        failures.append(f"input_ids mismatch: {ids[0].tolist()}")
    else:
        print("input_ids: byte-exact")

    aux = read_back("hidden_states", (1, n_prompt, len(LAYER_IDS) * HIDDEN), torch.bfloat16)
    aux = aux[0].float()
    print(f"aux: shape {tuple(aux.shape)}, rel_err vs engine rows = {rel_err(aux, ref_aux):.3e}")
    if aux.shape != ref_aux.shape or not torch.equal(aux, ref_aux):
        failures.append(
            f"aux payload differs from the engine's own rows (rel_err={rel_err(aux, ref_aux):.3e})"
        )

    last = read_back("target", (1, n_prompt, HIDDEN), torch.bfloat16)[0].float()
    print(f"last_hidden: shape {tuple(last.shape)}, finite={torch.isfinite(last).all().item()}")
    if not torch.isfinite(last).all():
        failures.append("last_hidden contains non-finite values")

    store.close()
    engine.shutdown()
    print("\n=== RESULT ===")
    if failures:
        for f in failures:
            print("FAIL:", f)
        sys.exit(1)
    print("VERIFY_E2E_OK")


if __name__ == "__main__":
    main()
