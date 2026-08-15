# coding=utf-8
"""Merge a trained MTP draft checkpoint back into the base target checkpoint.

The exporter owns checkpoint/index writing.  Architecture-specific knowledge
(target-side embed/lm_head key candidates, native key prefix) comes from the
registered MTP draft class — see ``specforge/modeling/draft/mtp/``.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
from typing import Dict, List, Optional, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from specforge.modeling.target.checkpoint import (
    load_selected_tensors,
    read_weight_map,
)


def _default_key_candidates() -> Tuple[List[str], List[str], str]:
    """Base-class defaults, imported lazily to keep this module import-light."""

    from specforge.modeling.draft.mtp.base import MTPDraftModel

    return (
        list(MTPDraftModel.TARGET_EMBED_KEY_CANDIDATES),
        list(MTPDraftModel.TARGET_HEAD_KEY_CANDIDATES),
        MTPDraftModel.NATIVE_KEY_PREFIX,
    )


def _resolve_key_candidates(
    mtp_checkpoint_path: str,
) -> Tuple[List[str], List[str], str]:
    """Return (embed candidates, head candidates, native prefix) for the draft.

    Reads the trained checkpoint's ``config.json`` and resolves its
    ``architectures[0]`` through the draft registry, so each MTP family can
    override its target-side key candidates on the draft class.
    """

    embed, head, prefix = _default_key_candidates()
    config_path = os.path.join(mtp_checkpoint_path, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                architectures = json.load(f).get("architectures") or []
            if architectures:
                from specforge.modeling.draft.registry import DRAFT_REGISTRY

                draft_cls = DRAFT_REGISTRY.get(architectures[0])
                if draft_cls is not None:
                    embed = list(
                        getattr(draft_cls, "TARGET_EMBED_KEY_CANDIDATES", embed)
                    )
                    head = list(
                        getattr(draft_cls, "TARGET_HEAD_KEY_CANDIDATES", head)
                    )
                    prefix = getattr(draft_cls, "NATIVE_KEY_PREFIX", prefix)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"  warning: could not resolve draft key candidates: {exc}")
    return embed, head, prefix


def convert_mtp_keys(
    state_dict: Dict[str, torch.Tensor], fmt: str, prefix: str = "mtp."
) -> Dict[str, torch.Tensor]:
    """Convert MTP weight keys to the requested output format.

    Training already saves the flat native layout that both SGLang and
    HF/vLLM MTP modules expect, so ``sglang`` and ``hf`` both return it
    unchanged; ``fmt`` is kept for backward compatibility. A legacy nested
    layout (``mtp.model.layers.0.*``) is normalized to flat.
    """

    converted = {}
    for k, v in state_dict.items():
        # Normalize legacy nested keys (mtp.model.layers.* -> mtp.layers.*).
        if k.startswith(f"{prefix}model.layers."):
            new_k = k.replace(f"{prefix}model.layers.", f"{prefix}layers.", 1)
        elif k == f"{prefix}model.norm.weight":
            new_k = f"{prefix}norm.weight"
        # Promote bare embed_tokens / lm_head saved by the training script to the
        # native namespace expected by vLLM/SGLang.
        elif k == "embed_tokens.weight":
            new_k = f"{prefix}embed_tokens.weight"
        elif k == "lm_head.weight":
            new_k = f"{prefix}lm_head.weight"
        else:
            new_k = k
        converted[new_k] = v
    return converted


def _find_base_key(state_dict: Dict[str, torch.Tensor], *candidates: str) -> str | None:
    """Return the first candidate key that exists in ``state_dict``."""

    for key in candidates:
        if key in state_dict:
            return key
    return None


def _copy_shared_embeddings(
    base_state: Dict[str, torch.Tensor],
    mtp_state: Dict[str, torch.Tensor],
    tie_word_embeddings: bool,
    embed_key_candidates: List[str],
    head_key_candidates: List[str],
    prefix: str,
) -> Dict[str, torch.Tensor]:
    """Copy base embed_tokens/lm_head into the MTP state if they are missing.

    During training the draft model typically shares ``embed_tokens`` and
    ``lm_head`` with the target model, so the saved MTP checkpoint does not
    contain those tensors.  vLLM/SGLang, however, instantiate their own
    ``mtp.embed_tokens`` (and a separate ``lm_head`` when weights are not tied),
    and expect them in the checkpoint.  Copying them from the base model keeps
    the merged checkpoint self-contained and avoids random-initialization of the
    MTP input/output embeddings at serving time.
    """

    embed_target = f"{prefix}embed_tokens.weight"
    head_target = f"{prefix}lm_head.weight"

    if embed_target not in mtp_state:
        embed_key = _find_base_key(base_state, *embed_key_candidates)
        if embed_key:
            mtp_state[embed_target] = base_state[embed_key]
            print(f"  copied {embed_key} -> {embed_target}")
        else:
            print(
                "  warning: base embed_tokens.weight not found; "
                f"{embed_target} will be randomly initialized"
            )

    if not tie_word_embeddings and head_target not in mtp_state:
        lm_head_key = _find_base_key(base_state, *head_key_candidates)
        if lm_head_key:
            mtp_state[head_target] = base_state[lm_head_key]
            print(f"  copied {lm_head_key} -> {head_target}")
        else:
            print(
                "  warning: base lm_head.weight not found; "
                f"{head_target} will be randomly initialized"
            )

    return mtp_state


def _patch_text_config(base_config: dict, draft_config: dict) -> dict:
    """Ensure base text_config contains MTP-critical dims from the draft config.

    Some Qwen3.5 base checkpoints omit ``head_dim`` in ``text_config``; vLLM's
    ``Qwen3_5TextConfig`` then falls back to its default (``head_dim=256``),
    which mismatches the trained MTP weights (e.g. q_norm/k_norm shape 128).
    We sync only the structural dims that must agree between base and draft.
    """

    keys_to_sync = [
        "head_dim",
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
    ]

    target = base_config
    if "text_config" in base_config:
        target = base_config["text_config"]

    source = draft_config
    if "text_config" in draft_config:
        source = draft_config["text_config"]

    for key in keys_to_sync:
        if key not in source:
            continue
        old = target.get(key)
        new = source[key]
        if old != new:
            target[key] = new
            print(f"  overriding text_config.{key}: {old} -> {new}")

    return base_config


def _load_first_checkpoint(checkpoint_dir: str) -> Dict[str, torch.Tensor]:
    """Load every tensor of a single-file checkpoint directory."""

    safetensors = glob.glob(os.path.join(checkpoint_dir, "*.safetensors"))
    bins = glob.glob(os.path.join(checkpoint_dir, "*.bin"))
    if safetensors:
        return load_selected_tensors(checkpoint_dir, lambda _key: True)
    if bins:
        return torch.load(bins[0], map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"No safetensors/bin weights found in {checkpoint_dir}")


def merge_mtp_into_base(
    base_model_path: str,
    mtp_checkpoint_path: str,
    output_path: str,
    key_format: str = "sglang",
) -> None:
    """Merge trained MTP weights into a copy of the base checkpoint.

    The output directory is a self-contained HF checkpoint loadable directly by
    SGLang's native MTP modules (no separate draft-model path).
    """

    os.makedirs(output_path, exist_ok=True)
    embed_key_candidates, head_key_candidates, prefix = _resolve_key_candidates(
        mtp_checkpoint_path
    )

    # Copy non-weight files from the base model so the output directory is a
    # fully self-contained HF checkpoint.
    for fname in os.listdir(base_model_path):
        src = os.path.join(base_model_path, fname)
        dst = os.path.join(output_path, fname)
        if os.path.isfile(src):
            shutil.copy2(src, dst)

    # Load MTP weights. The trainer saves them as a draft_model checkpoint:
    # model.safetensors + config.json + mtp.py.
    mtp_state = _load_first_checkpoint(mtp_checkpoint_path)
    mtp_state = convert_mtp_keys(mtp_state, key_format, prefix)

    # Determine whether word embeddings are tied so we know whether a separate
    # lm_head must be materialized for the MTP module.
    tie_word_embeddings = True
    base_config_path = os.path.join(base_model_path, "config.json")
    if os.path.exists(base_config_path):
        with open(base_config_path, "r") as f:
            base_cfg = json.load(f)
        # VLM checkpoints nest text config under "text_config".
        text_cfg = base_cfg.get("text_config", base_cfg)
        tie_word_embeddings = text_cfg.get("tie_word_embeddings", True)

    # Sharded base: write MTP weights into a dedicated shard and patch the index,
    # so the (large) base shards are never rewritten.
    weight_map = read_weight_map(base_model_path)
    index_files = glob.glob(os.path.join(base_model_path, "*.index.json"))
    if index_files:
        with open(index_files[0], "r") as f:
            index = json.load(f)
        weight_map = index.get("weight_map", weight_map)

        # If the base checkpoint already contains native MTP weights (e.g. an
        # official Qwen3.5 checkpoint), drop the old MTP entries so the trained
        # MTP weights replace them rather than duplicating them.
        old_mtp_keys = [k for k in weight_map if k.startswith(prefix)]
        for key in old_mtp_keys:
            del weight_map[key]
        if old_mtp_keys:
            print(
                f"Replaced {len(old_mtp_keys)} native MTP weight entries "
                "from base model."
            )

        # If the trained checkpoint did not save shared embeddings, copy them
        # from the base model shards so vLLM/SGLang can initialise the MTP
        # embed_tokens/lm_head from the merged checkpoint.
        embed_target = f"{prefix}embed_tokens.weight"
        head_target = f"{prefix}lm_head.weight"
        if embed_target not in mtp_state or (
            not tie_word_embeddings and head_target not in mtp_state
        ):
            base_state = load_tensors_for_merge(
                base_model_path,
                weight_map,
                embed_key_candidates + head_key_candidates,
            )
            mtp_state = _copy_shared_embeddings(
                base_state,
                mtp_state,
                tie_word_embeddings,
                embed_key_candidates,
                head_key_candidates,
                prefix,
            )

        # Write MTP weights into a dedicated shard.
        mtp_shard_name = "mtp-merged.safetensors"
        save_file(mtp_state, os.path.join(output_path, mtp_shard_name))

        # Update the weight map with the new MTP keys.
        for key in mtp_state.keys():
            weight_map[key] = mtp_shard_name

        # Save the updated index. Re-use the original index file name.
        index["weight_map"] = weight_map
        with open(
            os.path.join(output_path, os.path.basename(index_files[0])), "w"
        ) as f:
            json.dump(index, f, indent=2)
    else:
        # Single-file checkpoint: load base weights, merge, and rewrite.
        base_state = _load_first_checkpoint(base_model_path)
        base_safetensors = glob.glob(os.path.join(base_model_path, "*.safetensors"))
        base_bins = glob.glob(os.path.join(base_model_path, "*.bin"))
        out_name = os.path.basename(
            base_safetensors[0] if base_safetensors else base_bins[0]
        )

        # Remove any native MTP weights from the base state before overwriting
        # with the trained MTP weights.
        old_mtp_keys = [k for k in base_state if k.startswith(prefix)]
        for key in old_mtp_keys:
            del base_state[key]
        if old_mtp_keys:
            print(
                f"Replaced {len(old_mtp_keys)} native MTP weight entries "
                "from base model."
            )

        mtp_state = _copy_shared_embeddings(
            base_state,
            mtp_state,
            tie_word_embeddings,
            embed_key_candidates,
            head_key_candidates,
            prefix,
        )

        merged = {**base_state, **mtp_state}

        if out_name.endswith(".safetensors"):
            save_file(merged, os.path.join(output_path, out_name))
        else:
            torch.save(merged, os.path.join(output_path, out_name))

    # Ensure the merged config exposes the MTP structural dims.  vLLM/SGLang
    # use these values to build the MTP module; if the base config omits
    # ``head_dim`` (common for some Qwen3.5 checkpoints), the loader will use
    # its default and fail with a shape mismatch.
    draft_config_path = os.path.join(mtp_checkpoint_path, "config.json")
    output_config_path = os.path.join(output_path, "config.json")
    if os.path.exists(draft_config_path) and os.path.exists(output_config_path):
        with open(draft_config_path, "r") as f:
            draft_config = json.load(f)
        with open(output_config_path, "r") as f:
            base_config = json.load(f)
        patched_config = _patch_text_config(base_config, draft_config)
        with open(output_config_path, "w") as f:
            json.dump(patched_config, f, indent=2)

    # Copy over the MTP modeling file if present; some loaders need it for
    # trust_remote_code / auto_map resolution.
    mtp_py_src = os.path.join(mtp_checkpoint_path, "mtp.py")
    if os.path.exists(mtp_py_src):
        shutil.copy2(mtp_py_src, os.path.join(output_path, "mtp.py"))

    print(f"Merged checkpoint saved to {output_path}")
    print(f"  key format: {key_format}")
    print(f"  MTP tensors merged: {len(mtp_state)}")


def load_tensors_for_merge(
    base_model_path: str,
    weight_map: Dict[str, str],
    candidates: List[str],
) -> Dict[str, torch.Tensor]:
    """Load the given base-checkpoint tensors (only shards that hold them)."""

    base_state: Dict[str, torch.Tensor] = {}
    for candidate in candidates:
        if candidate not in weight_map:
            continue
        shard_path = os.path.join(base_model_path, weight_map[candidate])
        if not os.path.exists(shard_path):
            continue
        with safe_open(shard_path, framework="pt") as f:
            if candidate in f.keys():
                base_state[candidate] = f.get_tensor(candidate)
    return base_state


__all__ = ["convert_mtp_keys", "merge_mtp_into_base"]
