"""N0 smoke 1: plain (no-capture) forward / minimal generation on NPU."""

import sys

import sglang as sgl


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "/data1/f00538480/Qwen3.5-4B"
    engine = sgl.Engine(
        model_path=model_path,
        mem_fraction_static=0.5,
        page_size=64,
        disable_cuda_graph=True,
        mamba_ssm_dtype="bfloat16",
        log_level="info",
    )
    out = engine.generate(
        "The capital of France is",
        {"max_new_tokens": 8, "temperature": 0.0},
    )
    print("SMOKE_OUTPUT:", out["text"] if isinstance(out, dict) else out)
    engine.shutdown()
    print("SMOKE_OK")


if __name__ == "__main__":
    main()
