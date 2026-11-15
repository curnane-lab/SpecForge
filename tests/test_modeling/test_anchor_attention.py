"""Anchor SDPA must preserve visibility/gradients and bound saved intermediates."""

import unittest
from itertools import product
from unittest.mock import patch

import torch
from torch.nn import functional as F

from specforge.modeling.draft.anchor_attention import (
    AnchorBlockMask,
    anchor_block_attention,
)
from specforge.modeling.draft.dflash_mask import create_dflash_sdpa_mask
from specforge.modeling.draft.layers.attention.backends import (
    AttentionPreparationScope,
    SDPAAttentionBackend,
)
from specforge.modeling.draft.layers.attention.masks import AttentionLayout


def metadata(chunk=2, device="cpu"):
    anchors = torch.tensor([[0, 4, 9, 14, 16], [0, 3, 8, 0, 0]], device=device)
    keep = torch.tensor([[True] * 5, [True, True, True, False, False]], device=device)
    padding = (
        torch.arange(19, device=device)[None]
        < torch.tensor([19, 12], device=device)[:, None]
    )
    return AnchorBlockMask(anchors, keep, 19, 4, chunk, context_valid=padding)


def dense_mask(mask, causal=False, window=None):
    result = create_dflash_sdpa_mask(
        mask.anchors,
        mask.keep,
        mask.context_length,
        mask.block_size,
        mask.anchors.device,
        is_causal=causal,
        sliding_window=window,
    )
    if mask.context_valid is not None:
        result[..., : mask.context_length] &= mask.context_valid[:, None, None]
    return result


def dense_attention(q, k, v, mask, causal=False, window=None, dropout=0.0):
    active = mask.active_queries[:, None, :, None]
    allowed = dense_mask(mask, causal, window) | ~active
    result = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=allowed,
        enable_gqa=q.shape[1] != k.shape[1],
        dropout_p=dropout,
        scale=q.shape[-1] ** -0.5,
    )
    return result.masked_fill(~active, 0)


def qkv(mask, *, heads=4, kv_heads=2, value_dim=8, dtype=torch.float64):
    length = mask.anchors.shape[1] * mask.block_size
    return tuple(
        torch.randn(*shape, dtype=dtype, device=mask.anchors.device, requires_grad=True)
        for shape in (
            (2, heads, length, 8),
            (2, kv_heads, length + mask.context_length, 8),
            (2, kv_heads, length + mask.context_length, value_dim),
        )
    )


class AnchorAttentionTest(unittest.TestCase):
    def _parity(self, device="cpu", dtype=torch.float64):
        for causal, window, chunk, kv_heads, value_dim in product(
            (False, True),
            (None, 3),
            (1, 2, 9),
            (2, 4),
            (5, 8),
        ):
            with self.subTest(
                causal=causal,
                window=window,
                chunk=chunk,
                kv_heads=kv_heads,
                value_dim=value_dim,
            ):
                torch.manual_seed(7)
                mask = metadata(chunk, device)
                expected_inputs = qkv(
                    mask, kv_heads=kv_heads, value_dim=value_dim, dtype=dtype
                )
                inputs = tuple(
                    t.detach().clone().requires_grad_() for t in expected_inputs
                )
                expected = dense_attention(*expected_inputs, mask, causal, window)
                result = anchor_block_attention(
                    *inputs,
                    mask,
                    is_causal=causal,
                    sliding_window=window,
                    scale=8**-0.5,
                )
                probe = torch.randn_like(expected)
                expected_grads = torch.autograd.grad(
                    (expected * probe).sum(), expected_inputs
                )
                grads = torch.autograd.grad((result * probe).sum(), inputs)
                rtol, atol = (1e-10, 1e-10) if dtype == torch.float64 else (0.04, 0.03)
                for actual, reference in zip(
                    (result, *grads), (expected, *expected_grads)
                ):
                    torch.testing.assert_close(actual, reference, rtol=rtol, atol=atol)
                self.assertTrue(
                    torch.equal(result[1, :, 12:], torch.zeros_like(result[1, :, 12:]))
                )

    def test_cpu_outputs_and_all_qkv_gradients(self):
        self._parity()

    def test_cpu_bfloat16_outputs_and_gradients(self):
        self._parity(dtype=torch.bfloat16)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_bfloat16_outputs_and_gradients(self):
        self._parity("cuda", torch.bfloat16)

    @unittest.skipUnless(
        hasattr(torch, "npu") and torch.npu.is_available(), "Ascend is required"
    )
    def test_npu_bfloat16_outputs_and_gradients(self):
        self._parity("npu", torch.bfloat16)

    def test_all_inactive_rows_have_zero_output_and_gradients(self):
        mask = metadata()
        mask.keep.zero_()
        inputs = qkv(mask)
        result = anchor_block_attention(
            *inputs, mask, is_causal=False, sliding_window=None, scale=8**-0.5
        )
        grads = torch.autograd.grad(result.sum(), inputs)
        for value in (result, *grads):
            self.assertTrue(torch.equal(value, torch.zeros_like(value)))

    def test_backward_does_not_retain_masks_or_per_tile_kv_copies(self):
        mask = metadata(chunk=1)
        inputs = qkv(mask)
        retained = []

        def save(tensor):
            retained.append(tensor)
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(save, lambda tensor: tensor):
            result = anchor_block_attention(
                *inputs, mask, is_causal=False, sliding_window=None, scale=8**-0.5
            )
        self.assertFalse(any(t.dtype == torch.bool for t in retained))
        original_storages = {t.untyped_storage().data_ptr() for t in inputs}
        self.assertTrue(
            all(
                t.numel() == 0 or t.untyped_storage().data_ptr() in original_storages
                for t in retained
            )
        )
        result.square().sum().backward()
        self.assertTrue(
            all(t.grad is not None and torch.isfinite(t.grad).all() for t in inputs)
        )

    def test_dropout_recomputation_replays_the_same_random_masks(self):
        mask = metadata()
        inputs = qkv(mask)
        expected_inputs = tuple(t.detach().clone().requires_grad_() for t in inputs)
        kwargs = dict(is_causal=True, sliding_window=7, scale=8**-0.5, dropout_p=0.25)
        torch.manual_seed(29)
        with patch(
            "specforge.modeling.draft.anchor_attention.checkpoint",
            lambda fn, *args, **kwargs: fn(*args),
        ):
            expected = anchor_block_attention(*expected_inputs, mask, **kwargs)
        expected_grads = torch.autograd.grad(expected.square().sum(), expected_inputs)
        torch.manual_seed(29)
        result = anchor_block_attention(*inputs, mask, **kwargs)
        grads = torch.autograd.grad(result.square().sum(), inputs)
        for actual, reference in zip((result, *grads), (expected, *expected_grads)):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    def test_canonical_backend_respects_layout_positions_and_active_queries(self):
        mask = metadata()
        queries = (mask.anchors[..., None] + torch.arange(mask.block_size)).flatten(1)
        keys = torch.cat(
            (torch.arange(mask.context_length).expand(2, -1), queries), dim=1
        )
        # Different logical distances exercise the backend's actual layout positions.
        query_positions, key_positions = queries * 2, keys * 2
        active = mask.active_queries.clone()
        active[:, 2] = False
        layouts = [
            AttentionLayout(
                query_positions=query_positions,
                key_positions=key_positions,
                visibility=visible,
                active_queries=active,
            )
            for visible in (dense_mask(mask), mask)
        ]
        inputs = qkv(mask)
        references = tuple(t.detach().clone().requires_grad_() for t in inputs)
        outputs = []
        for layout, tensors in zip(layouts, (references, inputs)):
            with AttentionPreparationScope() as scope:
                prepared = scope.prepare(
                    SDPAAttentionBackend(),
                    layout,
                    batch_size=2,
                    num_heads=4,
                    is_causal=True,
                    sliding_window=9,
                )
            output = prepared.attend(
                *tensors, scale=8**-0.5, dropout_p=0, training=True
            )
            outputs.append(output.mask_output(output.output.transpose(1, 2).flatten(2)))
        gradients = [
            torch.autograd.grad(out.square().sum(), tensors)
            for out, tensors in zip(outputs, (references, inputs))
        ]
        for actual, reference in zip(
            (outputs[1], *gradients[1]), (outputs[0], *gradients[0])
        ):
            torch.testing.assert_close(actual, reference, rtol=1e-10, atol=1e-10)

    def test_no_grad_uses_the_same_visibility(self):
        mask = metadata()
        inputs = qkv(mask)
        with (
            torch.no_grad(),
            patch(
                "specforge.modeling.draft.anchor_attention.checkpoint",
                side_effect=AssertionError,
            ),
        ):
            output = anchor_block_attention(
                *inputs, mask, is_causal=True, sliding_window=3, scale=8**-0.5
            )
            torch.testing.assert_close(output, dense_attention(*inputs, mask, True, 3))
