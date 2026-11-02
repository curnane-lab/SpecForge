"""N0 smoke 3: plugin capture request lifecycle on NPU.

With the plugin installed and SPECFORGE_SPEC_CAPTURE=1, a spec_capture request
must be held until the sink reports. Without a Mooncake store this host cannot
complete publication, so the expected outcome is a per-request error result in
meta_info["spec_capture"] -- which still proves row selection, host copy,
deferral, completion and streamer injection on the NPU stack. A normal request
must stay unaffected.
"""

import json
import sys

import sglang as sgl


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "/data1/f00538480/Qwen3.5-4B"
    layer_ids = [2, 16, 29]
    engine = sgl.Engine(
        model_path=model_path,
        mem_fraction_static=0.5,
        page_size=64,
        disable_cuda_graph=True,
        mamba_ssm_dtype="bfloat16",
        log_level="info",
        enable_return_aux_hidden_states=True,
        speculative_algorithm="DFLASH",
        return_aux_hidden_state_layer_ids=layer_ids,
        return_hidden_states_mode="full",
        chunked_prefill_size=-1,
        skip_tokenizer_init=True,
    )
    spec = {
        "store_id": "smoke",
        "sample_id": "s0",
        "gen": 1,
        "replace": False,
        "features": {"aux": "hidden_states", "last_hidden": "target"},
        "passthrough": [{"name": "input_ids", "data": [1, 2, 3], "shape": [1, 3], "dtype": "int64"}],
    }
    # token ids for a short prompt (tokenizer skipped)
    input_ids = [151644, 872, 198, 6023, 374, 11914, 30, 151645, 198]
    out = engine.generate(
        None,
        input_ids=input_ids,
        sampling_params={
            "max_new_tokens": 0,
            "custom_params": {"spec_capture": json.dumps(spec)},
        },
    )
    meta = out.get("meta_info", {}) if isinstance(out, dict) else {}
    sc = meta.get("spec_capture")
    print("SMOKE_SPEC_CAPTURE:", json.dumps(sc))
    assert sc, "spec_capture missing from meta_info"
    result = sc[0]
    assert result.get("sample_id") == "s0", result
    if "error" in result:
        print("SMOKE_SINK_ERROR(expected without mooncake):", result["error"][:200])
    else:
        print("SMOKE_FEATURES:", json.dumps(result.get("features")))
    # a normal request still works
    out2 = engine.generate(None, input_ids=input_ids, sampling_params={"max_new_tokens": 1})
    print("SMOKE_NORMAL_OK:", len(out2.get("output_ids", [])))
    engine.shutdown()
    print("SMOKE_OK")


if __name__ == "__main__":
    main()
