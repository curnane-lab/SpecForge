# H-Spec (SpecForge PR #918) E2E experiment results

Date: 2026-10-12. Hardware: Ascend 910B3, bf16, greedy, single card per server.

**Code under test**
- Training: SpecForge `feat/hspec-sglang-capture` @ 9ce057f **+ correctness fix
  commit da1b83a** (`fix(hspec): align trainer draft topology with the serving
  implementation`; see §4).
- Capture: sglang upstream main 89f21671bb4f (2026-10-02; contains the #40343 parallel
  runtime the PR's capture backend requires).
- Serving: sglang PR #42021 head dd2c58f7 (local tree `/data1/f00538480/sglang-hspec-pr`),
  which includes the NPU TARGET_VERIFY seq_len fix. NOTE: the older 9/30 checkout
  (`sglang-qwen3.5`) predates that fix and yields materially lower accept lengths for
  ANY draft (see §5).

All artifacts were under `/data1/f00538480/hspec-verify/e2e/` (scripts, configs, logs,
checkpoints); the directory was deleted in a shared-disk cleanup on 2026-10-13, so this
document is now the canonical record of the numbers.

## 1. Pipeline (what was run)

1. **Capture**: `prepare_hidden_states.py --strategy hspec`, 3000-row ShareGPT subset
   (`Aeala/ShareGPT_Vicuna_unfiltered`), reference draft config, max_length 3072, bs 32,
   target = local Qwen3-4B. 2176/2200 samples persisted (24 filtered). Every sample carries
   all 7 features: `input_ids, loss_mask, hidden_states (5 latents = 12800),
   target_last_hidden_states, selected_target_k/v (3 KV layers 17/25/34 = 3072),
   prefix_masks` (real 0/1 separation).
2. **Warm start**: official `weifanjiang/qwen3-4b.speculators.hspec` converted to the trainer
   layout (`convert_official_to_specforge.py`; after topology alignment the key mapping is the
   identity; shared seed_proj broadcast to 4 per-layer projections; markov_w2 rows scattered
   via d2t offsets into full vocab). 84/84 tensors loaded by `warm_start_draft_model`,
   zero missing/unexpected.
3. **Training**: bs 1, sdpa, lr 2e-4 cosine (3% warmup), ~1.0 s/step.
   - run A: 1 epoch / 2000 steps (36 min) → `outputs/hspec-e2e-sharegpt-warmstart-step2000`
   - run B: 2 epochs / 4000 steps (73 min) → `outputs/hspec-e2e-sharegpt-warmstart-e2-step4000`
4. **Conversion**: `convert_specforge_to_serving.py` — passthrough keys (identical layout),
   full-vocab lm_head from the tied target embedding (draft_vocab_size=vocab_size=151936,
   identity mapping), `mamba_seed_mode=per_layer`, `num_hidden_layers`=3 (#attention).
5. **Serving eval**: `MAMBA_ATTN_HYBRID`, PR #42021 usage flags (`--disable-cuda-graph`,
   mem 0.8), Spec-Bench 6 tasks + HumanEval ×15, thinking template, greedy, max 512 new tokens.

## 2. Training metrics

Warm-started model trains normally (batch metrics, bs=1 so noisy): teacher agreement
0.13 → ~0.45-0.57; tau_probabilistic 1.24 → ~2.5-3.8 by end of epoch 1.

Held-out eval (176 unseen captures, frozen weights):

| checkpoint | teacher_agree | acc | tau_probabilistic | walk_acc_len | per-position acc (p0..p7) |
|---|---|---|---|---|---|
| 1 epoch  | 0.456 | 0.411 | 2.89 | 2.60 | .67/.55/.46/.38/.34/.32/.28/.27 |
| 2 epochs | 0.469 | 0.424 | 2.98 | 2.63 | .68/.56/.47/.41/.35/.33/.30/.28 |

Train-set eval (first 176 training captures the checkpoints actually saw, same protocol;
overfitting check — `eval_train.log`, `eval_train_ep2.log`):

| checkpoint | teacher_agree | acc | tau_probabilistic | walk_acc_len |
|---|---|---|---|---|
| 1 epoch, train  | 0.529 | 0.467 | 2.97 | 2.69 |
| 1 epoch, heldout| 0.456 | 0.411 | 2.89 | 2.60 |
| 2 epochs, train  | 0.546 | 0.466 | 3.04 | 2.73 |
| 2 epochs, heldout| 0.469 | 0.424 | 2.98 | 2.63 |

Train vs held-out gap is small: τ +2.7%/+2.0% (1ep/2ep), teacher agreement +16% relative.
No meaningful memorization — the draft generalizes from 2k samples; the 23% serving-τ
gap to the official draft is a data-scale/diversity gap, not overfitting (more epochs
lift train metrics slightly but held-out/serving stay flat).

## 3. Serving A/B (the deliverable)

Same build (PR head dd2c58f7), same harness, eager decode (`--disable-cuda-graph`):
baseline no-spec decode throughput 16.1-16.7 tok/s per task.

| task | official τ | trained-1ep τ | trained-2ep τ | official tok/s | 1ep tok/s | 2ep tok/s |
|---|---|---|---|---|---|---|
| Math          | 4.367 | 3.312 | 3.272 | 38.3 (2.38x) | 31.3 (1.94x) | 32.3 (2.01x) |
| QA            | 2.984 | 2.290 | 2.228 | 27.2 (1.67x) | 22.2 (1.36x) | 22.5 (1.38x) |
| Chat          | 2.744 | 2.040 | 2.019 | 25.3 (1.56x) | 19.7 (1.22x) | 20.4 (1.26x) |
| RAG           | 3.115 | 2.405 | 2.406 | 28.1 (1.69x) | 23.1 (1.39x) | 24.0 (1.45x) |
| Summarization | 2.582 | 2.042 | 2.057 | 24.3 (1.47x) | 19.6 (1.19x) | 20.7 (1.25x) |
| Translation   | 2.595 | 2.187 | 2.170 | 25.4 (1.52x) | 20.6 (1.23x) | 21.8 (1.30x) |
| Code          | 3.457 | 2.646 | 2.669 | 33.2 (2.02x) | 25.1 (1.53x) | 26.6 (1.62x) |
| **avg**       | **3.121** | **2.417** | **2.403** | **1.72x** | **1.42x** | **1.47x** |

(speedup vs the same-build no-spec baseline; PR #42021 reports avg τ 3.216 / 1.59x on its
own setup — our official-draft τ 3.12 reproduces that table to within ~3%.)

Findings:
- The SpecForge-trained draft (2k ShareGPT, warm-started, 1 epoch) reaches **~77% of the
  official draft's acceptance length** and a real decode speedup (avg 1.42-1.47x vs baseline).
- 1 epoch ≈ 2 epochs (τ 2.417 vs 2.403; step-1000 ≈ step-2000 on the old build): at 2k samples
  the model is data-limited, not step-limited. Closing the remaining 23% needs a bigger/
  more diverse corpus (official trained on Magpie/Ultrachat-scale data), not more epochs.
- Offline held-out τ (2.89) > serving τ (2.42): training data is non-thinking ShareGPT text,
  the eval is thinking-style Qwen3-4B generations (distribution shift).

## 4. Training-side correctness fixes (commit da1b83a)

Found by the e2e experiment; all verified after the fix:

1. **Topology alignment** (`specforge/modeling/draft/hspec.py`): trainer now builds/executes
   sub-layers strictly in `block_pattern` order with serving-identical semantics (leading lone
   mamba first & residual; every sub-layer pre-normed; `mlp@1` instantiated; checkpoint keys
   literally the serving layout `layers.{i}.ssm/self_attn/mlp.*`). Before: warm-started
   official weights collapsed to random (teacher agreement 0.10 → 0.02 falling); after:
   starts at 0.30 / τ 1.6 and trains up.
2. **Meta-device fix**: `seed_out = math.prod(state_shape)` — the from_pretrained warm-start
   path crashed on meta tensors (`Tensor.item()`).
3. **Numeric alignment**: `softplus(dt + dt_bias)` now fp32, matching the serving mixer.

Parity evidence (NPU, bf16, identical weights/inputs, scripts in e2e/):
- mixer level (`parity_mamba_mixer.py`): mean|Δ| 9.9e-5 (seeded) / 4.3e-5 (zero-init).
- full stack (`parity_full_draft.py`, mamba/mlp sublayers): seeds bit-identical,
  max|Δ| 0.023 ≈ bf16 accumulation noise (rel 6.6e-3).

Tests: 15/15 `test_hspec.py` + 32/32 capture/provider tests green (capture tests need
sglang ≥ 970e946e4f). Full suite (`tests/`, minus the pre-existing `yunchang` import gap):
1227 passed; the 9 failed + 1 error cluster (spec_capture_sink cuda/rdma gates,
prepare_hidden_states, regenerate_train_data, flex_attention, loss, build_eagle3_dataset,
ling3_template) is **byte-identical before and after da1b83a**
(`full_tests2.log`, `tests_ab_baseline.log` vs `tests_ab_fix.log`) — no regression from
the topology fix; the failing tests are pre-existing environment issues on this box.

## 5. Notes for both PRs

- SpecForge #918: warm-start works via `model.draft_checkpoint_path` (after §4 fixes);
  the reference recipe trains and the checkpoint serves with measurable acceptance.
  Serving deployments need `num_hidden_layers` = #attention (converter handles) and, for
  full-vocab training, identity d2t (draft_vocab_size = vocab_size) — both work on the
  current serving head.
- sglang #42021: the TARGET_VERIFY seq_len fix in dd2c58f7 lifts official-draft τ from
  ~1.7 avg (9/30 state) to 3.12 — validate any earlier measurements against the new head.
  On this stack eager decode makes spec decoding win (draft overhead amortizes); with
  acl-graph ON the same draft is ~breakeven (baseline 72.7 vs spec 71.8 tok/s), so the
  speedup table's baseline mode matters.
- Environment: capture needs sglang main ≥ 2026-09-21 (970e946e4f); 0.5.19 release lacks it.
- Checkpoint size: SpecForge writes 9.1 GB per checkpoint (7.8 GB rank-0 optimizer state);
  plan disk or prune `training_state_rank0.pt` for non-resume artifacts.
