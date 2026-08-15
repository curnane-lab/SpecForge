# coding=utf-8
"""Model-agnostic selective loading from local or Hugging Face checkpoints.

These helpers know nothing about any model family or key naming convention;
callers provide the keys or the predicate.  Both sharded checkpoints
(``*.safetensors.index.json``) and single-file checkpoints are supported.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Callable, Dict, Iterable, List, Optional

import torch
from safetensors import safe_open


def resolve_checkpoint_dir(
    path_or_repo: str,
    cache_dir: Optional[str] = None,
    allow_patterns: Optional[List[str]] = None,
) -> str:
    """Return a local checkpoint directory, downloading from the Hub if needed."""

    if os.path.exists(path_or_repo):
        return path_or_repo
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=path_or_repo,
        cache_dir=cache_dir,
        allow_patterns=allow_patterns or ["*.json", "*.safetensors", "*.bin"],
    )


def read_weight_map(checkpoint_dir: str) -> Dict[str, str]:
    """Return the ``weight_map`` of a sharded checkpoint, or {} if unsharded."""

    index_files = glob.glob(os.path.join(checkpoint_dir, "*.index.json"))
    if not index_files:
        return {}
    with open(index_files[0], "r") as f:
        index = json.load(f)
    return index.get("weight_map", {})


def list_checkpoint_keys(checkpoint_dir: str) -> List[str]:
    """List all tensor keys without loading tensor payloads."""

    weight_map = read_weight_map(checkpoint_dir)
    if weight_map:
        return sorted(weight_map.keys())
    for pattern in ("*.safetensors", "*.bin"):
        files = sorted(glob.glob(os.path.join(checkpoint_dir, pattern)))
        if files:
            target = files[0]
            if target.endswith(".safetensors"):
                with safe_open(target, framework="pt") as f:
                    return sorted(f.keys())
            state = torch.load(target, map_location="cpu", weights_only=True)
            return sorted(state.keys())
    raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")


def load_selected_tensors(
    checkpoint_dir: str,
    predicate: Callable[[str], bool],
) -> Dict[str, torch.Tensor]:
    """Load only the tensors whose key matches ``predicate``.

    Sharded checkpoints open just the shards that hold selected keys.
    """

    weight_map = read_weight_map(checkpoint_dir)
    selected: Dict[str, torch.Tensor] = {}
    if weight_map:
        shards = sorted({weight_map[k] for k in weight_map if predicate(k)})
        for shard in shards:
            shard_path = os.path.join(checkpoint_dir, shard)
            if not os.path.exists(shard_path):
                continue
            with safe_open(shard_path, framework="pt") as f:
                for key in f.keys():
                    if predicate(key):
                        selected[key] = f.get_tensor(key)
        return selected

    for pattern in ("*.safetensors", "*.bin"):
        files = sorted(glob.glob(os.path.join(checkpoint_dir, pattern)))
        if files:
            target = files[0]
            if target.endswith(".safetensors"):
                with safe_open(target, framework="pt") as f:
                    for key in f.keys():
                        if predicate(key):
                            selected[key] = f.get_tensor(key)
            else:
                state = torch.load(target, map_location="cpu", weights_only=True)
                for key, value in state.items():
                    if predicate(key):
                        selected[key] = value
            return selected
    raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")


def load_tensors_by_keys(
    checkpoint_dir: str, keys: Iterable[str]
) -> Dict[str, torch.Tensor]:
    """Load exactly ``keys`` (missing keys are simply absent from the result)."""

    wanted = set(keys)
    return load_selected_tensors(checkpoint_dir, lambda key: key in wanted)


__all__ = [
    "list_checkpoint_keys",
    "load_selected_tensors",
    "load_tensors_by_keys",
    "read_weight_map",
    "resolve_checkpoint_dir",
]
