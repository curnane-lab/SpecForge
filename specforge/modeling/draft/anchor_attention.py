"""DFlash SDPA over groups of independent anchor blocks."""

from dataclasses import dataclass
from functools import partial

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True, eq=False)
class AnchorBlockMask:
    """Compact context-prefix and block-local visibility for anchored proposals.

    Keys are ordered as context then all draft blocks. Each query sees context
    before its anchor and its own draft block. Layer causality and windows are
    applied separately. Keep this metadata immutable until backward completes.
    """

    anchors: Tensor
    keep: Tensor
    context_length: int
    block_size: int
    chunk_blocks: int
    context_valid: Tensor | None = None
    input_offset: int = 0
    context_offset: int = 0

    def __post_init__(self):
        for name in ("context_length", "block_size", "chunk_blocks"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            self.anchors.ndim != 2
            or not all(self.anchors.shape)
            or self.anchors.dtype not in (torch.int32, torch.int64)
            or self.keep.shape != self.anchors.shape
            or self.keep.dtype != torch.bool
            or self.keep.device != self.anchors.device
        ):
            raise ValueError(
                "anchors and keep must be aligned integer/bool [batch, blocks]"
            )
        if self.context_valid is not None and (
            self.context_valid.shape != (self.anchors.shape[0], self.context_length)
            or self.context_valid.dtype != torch.bool
            or self.context_valid.device != self.anchors.device
        ):
            raise ValueError("context_valid must be boolean [batch, context_length]")

    @property
    def active_queries(self):
        return self.keep.repeat_interleave(self.block_size, dim=1)

    def __call__(self, b, h, q, k):
        block = q // self.block_size
        limit = self.anchors[b, block] + self.input_offset + self.context_offset
        context = (k < self.context_length) & (k < limit)
        if self.context_valid is not None:
            context = (
                context & self.context_valid[b, k.clamp(max=self.context_length - 1)]
            )
        draft = (k >= self.context_length) & (
            (k - self.context_length) // self.block_size == block
        )
        return (context | draft) & self.keep[b, block]


def _attention_chunk(
    query,
    key,
    value,
    *,
    mask,
    start,
    end,
    is_causal,
    sliding_window,
    scale,
    dropout_p,
    positions,
):
    """Build tile-local masks and K/V inside the recomputation boundary."""
    context = mask.context_length
    device = query.device
    q_indices = torch.arange(start, end, device=device)
    key_indices = torch.cat((torch.arange(context, device=device), context + q_indices))
    batch = torch.arange(query.shape[0], device=device)[:, None, None]
    visible = mask(batch, None, q_indices[None, :, None], key_indices[None, None, :])
    if positions is None:
        query_positions = (
            mask.anchors[:, q_indices // mask.block_size]
            + mask.input_offset
            + q_indices % mask.block_size
        )
        key_positions = torch.cat(
            (
                torch.arange(context, device=device).expand(query.shape[0], -1),
                query_positions,
            ),
            dim=1,
        )
    else:
        query_positions = positions[0][:, start:end]
        key_positions = torch.cat(
            (
                positions[1][:, :context],
                positions[1][:, context + start : context + end],
            ),
            dim=1,
        )
    if is_causal:
        visible = visible & (query_positions[..., None] >= key_positions[:, None])
    if sliding_window is not None:
        distance = query_positions[..., None] - key_positions[:, None]
        visible = visible & (distance < sliding_window) & (distance > -sliding_window)
    active = mask.keep[:, start // mask.block_size : end // mask.block_size]
    active = active.repeat_interleave(mask.block_size, dim=1)[:, None, :, None]
    # Safe softmax on inactive rows, with zero outputs and gradients afterward.
    visible = visible[:, None] | ~active
    key = torch.cat(
        (key[:, :, :context], key[:, :, context + start : context + end]), dim=2
    )
    value = torch.cat(
        (value[:, :, :context], value[:, :, context + start : context + end]), dim=2
    )
    output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=visible,
        dropout_p=dropout_p,
        scale=scale,
        enable_gqa=query.shape[1] != key.shape[1],
    )
    return output.masked_fill(~active, 0)


def anchor_block_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: AnchorBlockMask,
    *,
    is_causal: bool,
    sliding_window: int | None,
    scale: float,
    dropout_p: float = 0.0,
    positions: tuple[Tensor, Tensor] | None = None,
) -> Tensor:
    """Attend without a full-sequence mask or cross-block draft K/V work.

    Each tile retains the shared context to avoid reading an anchor maximum
    back to the host. Checkpointing regenerates masks and concatenated K/V in
    backward, so autograd cannot retain their sum over all tiles. Dropout uses
    the checkpoint RNG state for replay. The caller supplies per-layer Q/K/V
    projections shared across tiles.
    """
    batch, blocks = mask.anchors.shape
    queries = blocks * mask.block_size
    if (
        any(t.ndim != 4 for t in (query, key, value))
        or query.shape[0] != batch
        or query.shape[2] != queries
        or key.shape[0] != batch
        or key.shape[2] != mask.context_length + queries
        or value.shape[:3] != key.shape[:3]
        or key.shape[-1] != query.shape[-1]
        or not key.shape[1]
        or query.shape[1] % key.shape[1]
        or any(t.device != mask.anchors.device for t in (query, key, value))
    ):
        raise ValueError(
            "anchor attention must match the context-then-draft Q/K/V layout"
        )
    if sliding_window is not None and sliding_window < 1:
        raise ValueError("sliding_window must be positive")
    value = value.to(query.dtype)
    recompute = torch.is_grad_enabled() and any(
        t.requires_grad for t in (query, key, value)
    )
    chunk_size = mask.chunk_blocks * mask.block_size
    outputs = []
    for start in range(0, queries, chunk_size):
        end = min(start + chunk_size, queries)
        attend = partial(
            _attention_chunk,
            mask=mask,
            start=start,
            end=end,
            is_causal=is_causal,
            sliding_window=sliding_window,
            scale=scale,
            dropout_p=dropout_p,
            positions=positions,
        )
        args = (query[:, :, start:end], key, value)
        outputs.append(
            checkpoint(attend, *args, use_reentrant=False)
            if recompute
            else attend(*args)
        )
    return torch.cat(outputs, dim=2)
