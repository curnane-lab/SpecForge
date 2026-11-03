"""Measure mixed-length capture publications with a delayed Mooncake sink."""

import asyncio
import json
import os
import sys

LAYER_IDS = [2, 16, 29]
HIDDEN = 2560
STORE_ID = "publication-memory"


def _spec(sample_id):
    return {
        "store_id": STORE_ID,
        "sample_id": sample_id,
        "gen": 1,
        "replace": False,
        "features": {"aux": "hidden_states", "last_hidden": "target"},
        "passthrough": [],
    }


async def _run(engine, lengths):
    async def fire(index, length):
        await asyncio.sleep(index * 0.75)
        input_ids = [(1000 + index * 17 + offset) % 150000 for offset in range(length)]
        started = asyncio.get_running_loop().time()
        for attempt in range(1, 31):
            output = await engine.async_generate(
                None,
                input_ids=input_ids,
                sampling_params={
                    "max_new_tokens": 0,
                    "custom_params": {"spec_capture": json.dumps(_spec(f"s{index}"))},
                },
            )
            result = output["meta_info"]["spec_capture"][0]
            if "error" not in result:
                break
            print(
                f"MEASURE retry sample=s{index} attempt={attempt} "
                f"error={result['error']}",
                flush=True,
            )
            await asyncio.sleep(0.5)
        elapsed = asyncio.get_running_loop().time() - started
        print(
            f"MEASURE response sample=s{index} tokens={length} attempts={attempt} "
            f"payload_mib={length * (len(LAYER_IDS) + 1) * HIDDEN * 2 / (1 << 20):.3f} "
            f"elapsed_s={elapsed:.3f} error={result.get('error')}",
            flush=True,
        )
        return result

    return await asyncio.gather(
        *(fire(index, length) for index, length in enumerate(lengths))
    )


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "/data1/f00538480/Qwen3.5-4B"
    lengths = [
        int(value)
        for value in os.environ.get(
            "MEASURE_LENGTHS", "4096,4096,4096,7,2048,129"
        ).split(",")
    ]
    os.environ["SPECFORGE_SPEC_CAPTURE"] = "1"
    os.environ["SGLANG_SPEC_CAPTURE_MAX_PENDING_BATCHES"] = "2"

    import sglang as sgl

    engine = sgl.Engine(
        model_path=model_path,
        base_gpu_id=int(os.environ.get("VERIFY_BASE_GPU_ID", "0")),
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
    try:
        results = asyncio.run(_run(engine, lengths))
    finally:
        engine.shutdown()
    assert all("error" not in result for result in results), results
    print("MEASURE_PUBLICATION_MEMORY_OK", flush=True)


if __name__ == "__main__":
    main()
