"""Frozen native-MTP alignment and independent HF decoder parity."""

import unittest

import torch
from transformers import Qwen3Config

from specforge.inference.mtp_capture import FrozenMTPCapture
from specforge.modeling.draft.mtp.qwen3_5 import Qwen3_5MTPDraftModel


def config():
    cfg = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        layer_types=["full_attention"],
        attn_output_gate=True,
        partial_rotary_factor=0.5,
    )
    cfg._attn_implementation = "sdpa"
    return cfg


class FrozenMTPCaptureTest(unittest.TestCase):
    def test_alignment_prefix_equivalence_and_no_future_leak(self):
        torch.manual_seed(9)
        teacher = FrozenMTPCapture(Qwen3_5MTPDraftModel(config()))
        ids = torch.tensor([1, 2, 3, 4, 5, 6])
        target = torch.randn(6, 16, requires_grad=True)
        full = teacher(ids, target)
        self.assertFalse(full.requires_grad)
        self.assertEqual(full[0].abs().sum().item(), 0)
        for end in range(2, 7):
            prefix = teacher(ids[:end], target[:end])
            torch.testing.assert_close(full[end - 1], prefix[-1], atol=1e-6, rtol=1e-5)
        changed = target.detach().clone()
        changed[3:] += 100
        torch.testing.assert_close(teacher(ids, changed)[:4], full[:4])

    def test_backbone_matches_independent_hf_qwen_layer(self):
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5DecoderLayer,
            Qwen3_5TextRotaryEmbedding,
        )

        torch.manual_seed(17)
        cfg = config()
        model = Qwen3_5MTPDraftModel(cfg).eval()
        native = Qwen3_5TextConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            layer_types=["full_attention"],
            attention_dropout=0.0,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.5,
                "mrope_section": [1, 1, 0],
            },
        )
        native._attn_implementation = "eager"
        reference = Qwen3_5DecoderLayer(native, layer_idx=0).eval()
        reference.load_state_dict(model.mtp.layers[0].state_dict(), strict=True)
        rotary = Qwen3_5TextRotaryEmbedding(native)
        ids, hidden = torch.tensor([[1, 2, 3, 4]]), torch.randn(1, 4, 16)
        position = torch.arange(4)[None]
        fused = model.mtp.fc(
            torch.cat(
                [
                    model.mtp.pre_fc_norm_embedding(model.embed_tokens(ids)),
                    model.mtp.pre_fc_norm_hidden(hidden),
                ],
                dim=-1,
            )
        )
        rope = rotary(fused, position.unsqueeze(0).expand(3, -1, -1))
        mask = torch.full((4, 4), float("-inf")).triu(1)[None, None]
        expected = model.mtp.norm(
            reference(fused, position_embeddings=rope, attention_mask=mask)
        )
        actual = model.forward_hidden(ids, hidden, position_ids=position)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
