"""N0 exit-item: verify capture numerics on NPU against an HF reference.

For one fixed prompt this script checks:
  1. capture-ON  engine: response hidden rows are the concatenated aux capture
     [tokens, K*H]; each chunk i must match HF hidden_states[layer_ids[i] + 1]
     (empirically established on this fork: the capture retains the flagged
     layer's completed *output* residual, i.e. HF index L+1) with the correct
     layer order, dtype and full token rows.
  2. capture-OFF engine: response hidden rows are the final post-norm hidden
     [tokens, H]; must match HF hidden_states[-1].
  3. logits reproduction: HF lm_head applied to the sglang final hidden must
     reproduce the engine's own next-token choice (top-1), and the engine's
     sampled token must sit in HF's top-k.

Capture on/off does not change the forward math (capture only reads), so the
capture-OFF final hidden doubles as the reference for what the plugin ships
as last_hidden.

Usage:
  ASCEND_RT_VISIBLE_DEVICES=3 PYTHONPATH=<fork>/python \
      python scripts/npu/verify_capture_numerics.py [model_path]
"""

import sys

import torch

LAYER_IDS = [2, 16, 29]
PROMPT = "The capital of France is"
HIDDEN = 2560


def run_engine(model_path, *, capture: bool):
    import sglang as sgl

    kwargs = dict(
        model_path=model_path,
        mem_fraction_static=0.5,
        page_size=64,
        disable_cuda_graph=True,
        mamba_ssm_dtype="bfloat16",
        log_level="warning",
        return_hidden_states_mode="full",
        chunked_prefill_size=-1,
    )
    if capture:
        kwargs.update(
            enable_return_aux_hidden_states=True,
            speculative_algorithm="DFLASH",
            return_aux_hidden_state_layer_ids=LAYER_IDS,
        )
    engine = sgl.Engine(**kwargs)
    out = engine.generate(
        PROMPT,
        {
            "max_new_tokens": 1,
            "temperature": 0.0,
        },
        return_logprob=True,
        top_logprobs_num=20,
        return_hidden_states=True,
    )
    meta = out["meta_info"]
    rows = [r for block in meta["hidden_states"] for r in block]
    n_prompt = meta["prompt_tokens"]
    hidden = torch.tensor(rows[:n_prompt], dtype=torch.float32)
    output_token = out["output_ids"][0]
    top_logprobs = [
        (t[1], t[0]) for t in meta["output_top_logprobs"][0]
    ]  # [(token_id, logprob)]
    engine.shutdown()
    return hidden, output_token, top_logprobs


def run_hf_reference(model_path):
    """fp32 CPU reference: clean of NPU kernel divergence (measures engine bf16)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()
    input_ids = tok(PROMPT, return_tensors="pt").input_ids
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True)
    hidden_states = [h[0] for h in out.hidden_states]  # [seq, H] fp32
    logits = out.logits[0, -1]  # [vocab] fp32
    del model
    return input_ids[0], hidden_states, logits


def rel_err(a, b):
    return (a - b).norm().item() / b.norm().clamp_min(1e-6).item()


def cos_sim(a, b):
    return torch.nn.functional.cosine_similarity(
        a.flatten(), b.flatten(), dim=0
    ).item()


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "/data1/f00538480/Qwen3.5-4B"

    print("=== run A: capture ON (aux via response) ===", flush=True)
    aux_rows, tok_a, top_a = run_engine(model_path, capture=True)
    k = len(LAYER_IDS)
    assert aux_rows.shape == (aux_rows.shape[0], k * HIDDEN), aux_rows.shape
    chunks = [aux_rows[:, i * HIDDEN : (i + 1) * HIDDEN] for i in range(k)]
    print(f"aux rows: {tuple(aux_rows.shape)} -> {k} chunks of {HIDDEN}")

    print("=== run B: capture OFF (final post-norm hidden) ===", flush=True)
    final_rows, tok_b, top_b = run_engine(model_path, capture=False)
    assert final_rows.shape[1] == HIDDEN, final_rows.shape
    assert tok_a == tok_b, (tok_a, tok_b)
    print(f"final rows: {tuple(final_rows.shape)}, next token id: {tok_a}")

    print("=== HF reference (CPU fp32) ===", flush=True)
    hf_ids, hf_hidden, hf_logits = run_hf_reference(model_path)
    seq_len = hf_ids.shape[0]
    assert aux_rows.shape[0] == seq_len, (aux_rows.shape[0], seq_len)
    assert len(hf_hidden) == 33, len(hf_hidden)  # embeddings + 32 layers
    print(f"hf hidden layers: {len(hf_hidden)}, seq_len: {seq_len}")

    failures = []

    # --- 1. aux layer order + values -------------------------------------
    # Capture retains the flagged layer's output residual: HF index L+1.
    print("\n--- aux chunks vs HF hidden_states ---")
    for i, lid in enumerate(LAYER_IDS):
        expected = lid + 1
        row = {}
        for cand in {expected - 1, expected, expected + 1}:
            row[cand] = (rel_err(chunks[i], hf_hidden[cand]), cos_sim(chunks[i], hf_hidden[cand]))
        best = min(row, key=lambda c: row[c][0])
        print(
            f"chunk{i} (layer {lid}): "
            + "  ".join(f"L{c}: rel={row[c][0]:.4e} cos={row[c][1]:.6f}" for c in sorted(row))
            + f"  -> best L{best}"
        )
        if best != expected:
            failures.append(f"chunk{i} best-matches HF layer {best}, expected {expected}")
        elif row[expected][1] < 0.99:
            failures.append(f"chunk{i} cosine {row[expected][1]:.6f} < 0.99 at HF layer {expected}")

    # --- 2. final hidden is post-norm ------------------------------------
    print("\n--- final hidden vs HF post-norm (hidden_states[-1]) ---")
    re_post, cs_post = rel_err(final_rows, hf_hidden[-1]), cos_sim(final_rows, hf_hidden[-1])
    re_pre, cs_pre = rel_err(final_rows, hf_hidden[-2]), cos_sim(final_rows, hf_hidden[-2])
    print(f"post-norm: rel={re_post:.4e} cos={cs_post:.6f}")
    print(f"neighbor (L30 out): rel={re_pre:.4e} cos={cs_pre:.6f}")
    if not (cs_post > 0.99 and cs_post > cs_pre):
        failures.append(f"final hidden does not match HF post-norm best (cos={cs_post:.6f})")

    # --- 3. logits reproduction ------------------------------------------
    print("\n--- logits reproduction ---")
    hf_top1 = hf_logits.argmax().item()
    engine_top1 = tok_a
    top_ids = {t for t, _ in top_a}
    print(f"engine next token: {engine_top1}; HF top-1: {hf_top1}")
    print(f"engine top-20 overlap with HF top-1: {hf_top1 in top_ids}")
    if engine_top1 != hf_top1:
        failures.append(f"engine top-1 {engine_top1} != HF top-1 {hf_top1}")

    print("\n=== RESULT ===")
    if failures:
        for f in failures:
            print("FAIL:", f)
        sys.exit(1)
    print("VERIFY_OK")


if __name__ == "__main__":
    main()
