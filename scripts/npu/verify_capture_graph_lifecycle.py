"""N0 exit-item 3: NPU prefill graph replay must not corrupt a captured sample.

Design (per the v2 plan's graph-lifecycle gate):
  Phase 1 (eager reference): capture-ON engine with graphs disabled; record
    prompt A's aux rows.
  Phase 2 (graph): capture-ON engine with the prefill graph backend enabled
    (decode graph left at its default). Capture request A is submitted first,
    then a burst of normal requests and an identical capture A2 follow
    immediately via async_generate, so later prefill batches replay the graph
    while A's host copy/response pipeline is still draining.
  Pass iff:
    - the boot log proves prefill graphs were captured (the shell wrapper
      greps for "Capturing prefill shape"), and
    - A's aux rows match the eager reference within kernel tolerance, and
    - A2's aux rows match A's (a later replay of the same bucket must be
      deterministic).

Radix cache is disabled so A2 really re-runs a full prefill instead of
serving a prefix from cache.

Usage:
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  ASCEND_RT_VISIBLE_DEVICES=3 PYTHONPATH=<fork>/python:$PYTHONPATH \
      python scripts/npu/verify_capture_graph_lifecycle.py [model_path]
"""

import asyncio
import sys

import torch

LAYER_IDS = [2, 16, 29]
PROMPT_A = "The capital of France is"
BURST_PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "In a shocking finding, scientists discovered a herd of unicorns living in",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n",
    "The future of artificial intelligence is",
    "Once upon a time in a land far away",
]
HIDDEN = 2560


def boot(model_path, *, graphs: bool):
    import sglang as sgl

    kwargs = dict(
        model_path=model_path,
        mem_fraction_static=0.5,
        page_size=64,
        mamba_ssm_dtype="bfloat16",
        log_level="info",
        enable_return_aux_hidden_states=True,
        speculative_algorithm="DFLASH",
        return_aux_hidden_state_layer_ids=LAYER_IDS,
        return_hidden_states_mode="full",
        chunked_prefill_size=-1,
    )
    if graphs:
        kwargs.update(
            cuda_graph_backend_prefill="breakable",
            cuda_graph_bs_prefill=[64, 128, 512],
            disable_radix_cache=True,
        )
    else:
        kwargs["disable_cuda_graph"] = True
    return sgl.Engine(**kwargs)


def rows_of(out):
    meta = out["meta_info"]
    n_prompt = meta["prompt_tokens"]
    rows = [r for block in meta["hidden_states"] for r in block]
    assert len(rows) >= n_prompt, (len(rows), n_prompt)
    t = torch.tensor(rows[:n_prompt], dtype=torch.float32)
    assert t.shape[1] == len(LAYER_IDS) * HIDDEN, t.shape
    return t


def rel_err(a, b):
    return (a - b).norm().item() / b.norm().clamp_min(1e-6).item()


def cos_sim(a, b):
    return torch.nn.functional.cosine_similarity(
        a.flatten(), b.flatten(), dim=0
    ).item()


async def phase2(engine):
    async def fire(prompt, capture, ntok):
        return await engine.async_generate(
            prompt,
            {"max_new_tokens": ntok, "temperature": 0.0},
            return_hidden_states=capture,
        )

    # A leads; the burst and A2 follow in the same event-loop turn so the
    # scheduler sees them as closely spaced batches.
    tasks = [asyncio.create_task(fire(PROMPT_A, True, 1), name="A")]
    for j, p in enumerate(BURST_PROMPTS):
        tasks.append(asyncio.create_task(fire(p, False, 8), name=f"burst{j}"))
    tasks.append(asyncio.create_task(fire(PROMPT_A, True, 1), name="A2"))
    outs = await asyncio.gather(*tasks)
    return rows_of(outs[0]), rows_of(outs[-1])


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "/data1/f00538480/Qwen3.5-4B"
    failures = []

    print("=== phase 1: eager reference ===", flush=True)
    engine = boot(model_path, graphs=False)
    ref_a = rows_of(
        engine.generate(
            PROMPT_A,
            {"max_new_tokens": 1, "temperature": 0.0},
            return_hidden_states=True,
        )
    )
    engine.shutdown()
    print(f"eager rows for A: {tuple(ref_a.shape)}")

    print("=== phase 2: prefill graph + interleaved replays ===", flush=True)
    engine = boot(model_path, graphs=True)
    graph_a, graph_a2 = asyncio.run(phase2(engine))
    engine.shutdown()

    print("\n--- A(graph) vs eager reference ---")
    for i, lid in enumerate(LAYER_IDS):
        a = graph_a[:, i * HIDDEN : (i + 1) * HIDDEN]
        r = ref_a[:, i * HIDDEN : (i + 1) * HIDDEN]
        print(f"chunk{i} (layer {lid}): rel={rel_err(a, r):.4e} cos={cos_sim(a, r):.6f}")
        if not (rel_err(a, r) < 2e-2 and cos_sim(a, r) > 0.999):
            failures.append(f"A chunk{i} diverges from eager under graph replay")

    print("--- A2(graph, after burst) vs A(graph) ---")
    for i, lid in enumerate(LAYER_IDS):
        a = graph_a[:, i * HIDDEN : (i + 1) * HIDDEN]
        b = graph_a2[:, i * HIDDEN : (i + 1) * HIDDEN]
        print(f"chunk{i} (layer {lid}): rel={rel_err(a, b):.4e} cos={cos_sim(a, b):.6f}")
        if not (rel_err(a, b) < 2e-2 and cos_sim(a, b) > 0.999):
            failures.append(f"A2 chunk{i} diverges from A; replay may corrupt capture")

    print("\n=== RESULT ===")
    if failures:
        for f in failures:
            print("FAIL:", f)
        sys.exit(1)
    print("VERIFY_GRAPH_OK")


if __name__ == "__main__":
    main()
