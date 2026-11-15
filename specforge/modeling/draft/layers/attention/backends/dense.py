"""Dense mask preparation shared by eager and SDPA execution."""

from dataclasses import dataclass

from torch import Tensor
from torch.nn import functional as F

from ....anchor_attention import AnchorBlockMask, anchor_block_attention
from ..masks import dense_attention_mask
from .base import AttentionBackend
from .eager import eager_attention


@dataclass(frozen=True, eq=False)
class DenseMask:
    visibility: Tensor
    empty_rows: Tensor


@dataclass(frozen=True, eq=False)
class AnchorMask:
    blocks: AnchorBlockMask
    positions: tuple[Tensor, Tensor]
    is_causal: bool
    sliding_window: int | None


class DenseAttentionBackend(AttentionBackend):
    def prepare(self, layout, *, batch_size, num_heads, is_causal, sliding_window):
        if isinstance(layout.visibility, AnchorBlockMask):
            raise ValueError("anchor block attention requires the sdpa backend")
        mask = dense_attention_mask(
            layout,
            batch_size=batch_size,
            num_heads=num_heads,
            is_causal=is_causal,
            sliding_window=sliding_window,
        )
        visible_rows = mask.any(dim=-1, keepdim=True)
        active = visible_rows.squeeze(-1).any(dim=1).expand(batch_size, -1)
        return DenseMask(mask, ~visible_rows), active


class EagerAttentionBackend(DenseAttentionBackend):
    def attend(
        self,
        query,
        key,
        value,
        *,
        mask: DenseMask,
        scale,
        dropout_p,
        training,
        return_lse,
    ):
        result = eager_attention(
            query,
            key,
            value,
            attention_mask=mask.visibility,
            empty_rows=mask.empty_rows,
            scale=scale,
            dropout_p=dropout_p,
            training=training,
            return_lse=return_lse,
        )
        return result if return_lse else (result, None)


class SDPAAttentionBackend(DenseAttentionBackend):
    def prepare(self, layout, **options):
        if isinstance(layout.visibility, AnchorBlockMask):
            blocks = layout.visibility
            batch = options["batch_size"]
            queries = blocks.anchors.shape[1] * blocks.block_size
            if (
                blocks.anchors.shape[0] != batch
                or layout.query_positions.shape[1] != queries
                or layout.key_positions.shape[1] != blocks.context_length + queries
            ):
                raise ValueError(
                    "anchor block metadata must match the attention layout"
                )
            active = blocks.active_queries
            if layout.active_queries is not None:
                active = active & layout.active_queries
            return (
                AnchorMask(
                    blocks,
                    (
                        layout.query_positions.expand(batch, -1),
                        layout.key_positions.expand(batch, -1),
                    ),
                    options["is_causal"],
                    layout._resolve_window(options["sliding_window"]),
                ),
                active,
            )
        mask, active = super().prepare(layout, **options)
        # Prepare once per forward. Avoid undefined softmax/backward on empty rows.
        return DenseMask(mask.visibility | mask.empty_rows, mask.empty_rows), active

    def attend(
        self,
        query,
        key,
        value,
        *,
        mask: DenseMask | AnchorMask,
        scale,
        dropout_p,
        training,
        return_lse,
    ):
        if return_lse:
            raise NotImplementedError(
                "SDPA does not expose attention LSE; use eager or Flex"
            )
        if isinstance(mask, AnchorMask):
            return (
                anchor_block_attention(
                    query,
                    key,
                    value,
                    mask.blocks,
                    is_causal=mask.is_causal,
                    sliding_window=mask.sliding_window,
                    scale=scale,
                    dropout_p=dropout_p if training else 0.0,
                    positions=mask.positions,
                ),
                None,
            )
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask.visibility,
            scale=scale,
            dropout_p=dropout_p if training else 0.0,
            enable_gqa=query.shape[1] != key.shape[1],
        )
        return output.masked_fill(mask.empty_rows, 0), None
