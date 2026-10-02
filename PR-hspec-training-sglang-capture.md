# PR: H-Spec training with sglang offline capture

**Repo:** curnane-lab/SpecForge
**Branch:** `feat/hspec-sglang-capture` (single commit `411cc75` on `main` `3cb0510`)
**Diff:** 23 files, +2897 / −254

---

## Title

```
feat(specforge): add H-Spec (mamba_attn_hybrid) training with sglang offline capture
```

## Body

### What

Add the H-Spec training side to SpecForge: a hybrid Mamba/target-KV speculative-draft
objective that matches the serving implementation in
[sgl-project/sglang#42021](https://github.com/sgl-project/sglang/pull/42021)
(H-Spec, `mamba_attn_hybrid` over Qwen3-4B), together with an offline capture
backend wired to the current sglang parallel API.

### Draft architecture (`specforge/modeling/draft/hspec.py`)

- Hybrid Mamba-attention-MLP block pattern; each attention block borrows
  post-RoPE target K/V from a configured target layer.
- **Split-source attention**: exact fp32 log-sum-exp merge of the borrowed
  target prefix and the transient local draft block.
  - Per-query context visibility: each anchor block only sees target tokens
    before its anchor (no future-token leak).
  - Same-block bidirectional draft masks, matching the serving ENCODER_ONLY
    infill semantics.
  - The DFlash attention mask is consumed end to end; an uninterpretable mask
    fails fast instead of silently degrading to an all-visible prefix.
  - `merge_attention_states` guards fully-masked rows (invalid blocks) against NaN.
- **Mamba mixer aligned with the serving `HSpecMambaMixer`**: biased causal
  convolution evaluated in fp32 (weights included), gated RMSNorm applied after
  the gate, softplus dt without a clamp floor, per-layer `seed_projs` seeded
  from each block anchor's fused latent (`hidden_norm(fc(latent))`), per-block
  recurrences with no cross-layer state chaining.
- No `hidden_size == mamba_num_heads * mamba_head_dim` coupling: the official
  44×64 mapping (2816 inner width vs hidden 2560) instantiates directly, with
  `out_proj` mapping back to the hidden size.
- The base uniform decoder stack is deleted: the hybrid/partial stacks replace
  it, so DDP with `find_unused_parameters=False` trains cleanly.

### Offline capture backend (`specforge/offline_capture/sglang_backend/`)

- Ported to the new sglang parallel API (`970e946e4f`, "Retire the per-runner
  parallel record"): the runner no longer takes a `ParallelState`; the backend
  publishes the scheduler-role config and builds the parallel runtime through
  `bootstrap.init_parallel_runtime` / `init_layer_runtime`, mirroring the
  upstream one-batch benchmark entry point.
- Real paged-pool post-RoPE K/V capture with an NPU device guard for
  `npu_format_cast`; `diag` prints removed.
- Persists **seven** features per sample, including the new **`prefix_masks`**
  (1 = real context token, 0 = padding) so padded target K/V stay out of the
  prefix softmax.

### Data / training plumbing

- `prefix_masks` flows through the offline reader → normalizer → collator →
  `HSpecTrainStrategy` → draft forward; the hspec offline feature contract
  requires it, and the collator zero-pads it (0 = invisible).
- `training.attention_backend` now defaults to `None` and resolves from the
  selected algorithm's capability set (hspec: eager/sdpa), so restricted
  algorithms are not forced through the global `flex_attention` default.
- `--attention-backend` in `scripts/prepare_hidden_states.py` follows the same
  resolution; hspec capture runs fail fast when the backend yields no feature
  rows.

### Configs / examples

- `configs/qwen3-4b-hspec-reference.json`: translated from the official
  speculators checkpoint (weifanjiang/qwen3-4b.speculators.hspec) — KV layers
  17/25/34, latent fusion 1/9/17/25/34 — with a colocated offline recipe.
- `configs/qwen3-4b-hspec-npu-debug.json`: minimal 2-sublayer debug topology
  for NPU smoke runs (marked `debug_only`).

### Testing

- Unit: mask semantics (anchor truncation, same-block bidirectionality, leak
  regression), split/cat parity under block masks, mixer recurrence vs a
  stepwise reference, NPU parity (gated), plus an integration test driving
  `OnlineHSpecModel.forward → _forward_draft_blocks (real 4-D mask) → draft
  forward → loss → backward` with `seed_projs` gradient checks. 160 tests green.
- End-to-end on Ascend NPU (Qwen3-4B):
  - Capture: `prepare_hidden_states.py --strategy hspec` over real conversations,
    4/4 samples, all seven features persisted and consumed by the trainer data
    path untouched.
  - Multi-step training: real `OfflineManifestReader → LocalFeatureStore →
    FeatureDataLoader` batches → `HSpecTrainStrategy.forward_loss` → DDP-wrapped
    backward → optimizer steps; loss 2.41 → 1.75 over 6 steps with finite
    gradients and working `checkpoint_state_filter`.

### Notes / known limitations

- **H-Spec is offline-only by design**: it requires target post-RoPE K/V as
  training features; no streaming (colocated online) feature contract is
  registered. An online producer that exports K/V from a live server is future
  work.
- `num_hidden_layers` contract divergence: training derives it from the
  `block_pattern` length (all sub-layers) while the serving side requires it to
  equal the number of attention sub-layers; serving deployments need that field
  adjusted (documented in the reference config's `_notes`).
- The mixer conv is evaluated in fp32 to match the serving-side explicit
  tap-sum numerics; NPU parity is covered by the gated tests.

### Related

- Serving side: [sgl-project/sglang#42021](https://github.com/sgl-project/sglang/pull/42021)
- Reference implementation: weifanjiang/H-Spec
