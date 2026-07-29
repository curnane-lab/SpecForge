# VLM DFlash Support — Port Status (add_vl_support)

This branch ports sgl-project/SpecForge PR #585 (commit `9323a510`, author zyk42)
onto the server-only unified runtime. This document records what landed, what was
deliberately not ported, and what is still missing for end-to-end VLM DFlash
training.

## What this branch contains

- **Draft model (VLM-capable)** — `specforge/modeling/draft/dflash.py`:
  - `apply_rotary_pos_emb` supports partial rotation (`rotary_dim < head_dim`,
    required by Qwen3.5/Qwen3.6 with `partial_rotary_factor=0.25`).
  - `Qwen3InterleavedMultiRotaryEmbedding` + `mrope_interleaved` selection for
    Qwen3-VL style interleaved mRoPE position ids.
- **Draft configs** — `configs/qwen3-vl-8b-dflash-vlm-8layer.json`,
  `configs/qwen3-vl-30b-a3b-dflash-vlm-8layer.json`,
  `configs/qwen3.5-9b-dflash-vlm-8layer.json`,
  `configs/qwen3.5-35b-a3b-dflash-vlm-8layer.json`
  (mRoPE sections, `partial_rotary_factor`, per-family `target_layer_ids`;
  Qwen3-VL skips the first 3 deepstack layers).
- **Weight-key resolution** — `specforge/modeling/target/target_utils.py`:
  `QWEN3_VL_MODEL_TYPES` + `resolve_target_weight_keys()`;
  `TargetEmbeddingsAndHead.from_pretrained` auto-selects
  `model.language_model.embed_tokens.weight` for VLM targets when the embed key
  is unset or left at the LLM default. Explicit keys are honored.
  `populate_dflash_generated_config` reads language-model depth via the
  `text_config` fallback.

## Not ported (by design)

PR #585 targeted the pre-#678 script/HF-backend stack, which no longer exists on
main. The following were **not** carried over:

- HF-backend VLM capture (`specforge/modeling/target/dflash_target_model.py`,
  ~580 lines: `_build_vlm_reqs`, `MRotaryEmbedding.get_rope_index`,
  `mm_token_type_ids` generation, pixel_values slicing).
- `scripts/train_dflash.py` VLM plumbing (`--is-vlm`, processor loading,
  mixed text+VLM batches at `batch_size=1`).
- `QwenVLOnlineDFlashModel` wiring — note that PR #585 **referenced this class
  but never defined it** (it lived in the author's private stack).
- Two accidental reverts in the original diff (domino projector code, D-PACE
  CLI args) were dropped during the cherry-pick.

## Missing for end-to-end VLM training

**Server-side multimodal capture.** Online capture now runs on an external
SGLang server (spec-capture patch + Mooncake transport). Training a VLM DFlash
additionally needs:

1. The capture server to accept multimodal requests (pixel values / image
   tokens) and produce aux hidden states + mRoPE position ids for them.
2. A `modality="multimodal"` `FeatureContract` plus collator/schema support in
   `specforge/algorithms/*` (today every built-in algorithm declares
   `modality="text"` only).

Until then this branch is the **foundation layer**: draft model + configs +
weight-key resolution. Text-only DFlash/Domino training (including on Ascend
NPU, via the merged `npu_disaggregated` work) is unaffected.

## Reference results from PR #585 (HF stack, author-validated)

- Qwen3-VL-30B-A3B-Thinking, 278K target-regenerated samples, 5-layer draft,
  block_size=8: accept length 3.52, +35.8% inference speedup (4x RTX 5090,
  TP=4, SGLang 0.5.12).
- Data must be target-model greedy-regenerated; system prompt must match
  between training and inference; <10K samples overfit severely.
