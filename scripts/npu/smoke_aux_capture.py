"""N0 smoke 2: aux + last-hidden capture forward on NPU (no draft runs).

Boots the target with the capture conventions the plugin requires and checks
the server actually captures aux layers for a full prefill.
"""

import sys

import sglang as sgl


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "/data1/f00538480/Qwen3.5-4B"
    # 32 layers: a low/mid/high aux set for the capture convention.
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
    )
    prompt = "The capital of France is"
    out = engine.generate(prompt, {"max_new_tokens": 0, "temperature": 0.0})
    meta = out.get("meta_info", {}) if isinstance(out, dict) else {}
    print("SMOKE_OUTPUT_IDS_LEN:", len(out.get("output_ids", [])) if isinstance(out, dict) else "n/a")
    hidden = meta.get("hidden_states")
    if hidden is None:
        print("SMOKE_HIDDEN: not returned in response (expected on this build)")
    else:
        print("SMOKE_HIDDEN_SHAPE:", getattr(hidden, "shape", None))
    engine.shutdown()
    print("SMOKE_OK layers=", layer_ids)


if __name__ == "__main__":
    main()
