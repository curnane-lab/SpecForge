import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from specforge.inference.mtp_capture import (
    FrozenMTPCapture,
    _native_config,
    _NativeMTPModel,
)


def tiny_config():
    config = Qwen3_5TextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        layer_types=["full_attention"],
        mtp_num_hidden_layers=1,
        rope_parameters={"rope_type": "default", "partial_rotary_factor": 0.5},
    )
    config._attn_implementation = "sdpa"
    return config


def native_weights(model):
    names = {
        "embedding_norm.weight": "mtp.pre_fc_norm_embedding.weight",
        "hidden_norm.weight": "mtp.pre_fc_norm_hidden.weight",
        "fc.weight": "mtp.fc.weight",
        "text.embed_tokens.weight": "model.language_model.embed_tokens.weight",
        "text.norm.weight": "mtp.norm.weight",
    }
    return {
        names.get(
            key, key.replace("text.layers.0.", "mtp.layers.0.")
        ): value.contiguous()
        for key, value in model.state_dict().items()
    }


def npu_available():
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return False
    return hasattr(torch, "npu") and torch.npu.is_available()


class DMTPCaptureTest(unittest.TestCase):
    def test_checkpoint_alignment_and_frozen_inputs(self):
        torch.manual_seed(17)
        model = _NativeMTPModel(tiny_config())
        with TemporaryDirectory() as directory:
            root = Path(directory)
            save_file(native_weights(model), str(root / "model.safetensors"))
            (root / "config.json").write_text(json.dumps(tiny_config().to_dict()))
            restored = FrozenMTPCapture.from_pretrained(
                root, device="cpu", dtype=torch.float32
            )
        ids = torch.arange(1, 7)
        target = torch.randn(6, 16, requires_grad=True)
        actual = restored(ids, target)
        expected = FrozenMTPCapture(model)(ids, target)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertFalse(actual.requires_grad)
        self.assertIsNone(target.grad)
        self.assertEqual(actual[0].abs().sum().item(), 0)
        for end in range(2, 7):
            torch.testing.assert_close(
                actual[end - 1], restored(ids[:end], target[:end])[-1]
            )
        changed = target.detach().clone()
        changed[3:] += 100
        torch.testing.assert_close(restored(ids, changed)[:4], actual[:4])
        with self.assertRaisesRegex(ValueError, "lengths differ"):
            restored(ids[:-1], target)
        self.assertTrue(all(not p.requires_grad for p in restored.model.parameters()))

    @unittest.skipUnless(npu_available(), "native MTP NPU test requires Ascend NPU")
    def test_npu_capture_and_prefix_equivalence(self):
        torch.manual_seed(19)
        model = _NativeMTPModel(tiny_config())
        with TemporaryDirectory() as directory:
            root = Path(directory)
            save_file(native_weights(model), str(root / "model.safetensors"))
            (root / "config.json").write_text(json.dumps(tiny_config().to_dict()))
            restored = FrozenMTPCapture.from_pretrained(
                root, device="npu", dtype=torch.bfloat16
            )
        ids = torch.arange(1, 7)
        target = torch.randn(6, 16, device="npu", dtype=torch.bfloat16)
        output = restored(ids, target)
        self.assertEqual(output.device.type, "npu")
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(output[0].abs().sum().item(), 0)
        for end in range(2, 7):
            torch.testing.assert_close(
                output[end - 1], restored(ids[:end], target[:end])[-1]
            )

    def test_unsupported_native_mtp_is_rejected(self):
        text = SimpleNamespace(
            model_type="qwen3_5_text",
            num_experts=0,
            mtp_num_hidden_layers=2,
            mtp_use_dedicated_embeddings=False,
        )
        with patch(
            "specforge.inference.mtp_capture.AutoConfig.from_pretrained",
            return_value=text,
        ):
            with self.assertRaisesRegex(ValueError, "one native MTP layer"):
                _native_config("unused")
            text.mtp_num_hidden_layers = 1
            text.quantization_config = {"quant_method": "int8"}
            with self.assertRaisesRegex(ValueError, "unquantized"):
                _native_config("unused")

    def test_incomplete_checkpoint_is_not_silently_initialized(self):
        model = _NativeMTPModel(tiny_config())
        weights = native_weights(model)
        del weights["mtp.norm.weight"]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            save_file(weights, str(root / "model.safetensors"))
            (root / "config.json").write_text(json.dumps(tiny_config().to_dict()))
            with self.assertRaisesRegex(KeyError, "mtp.norm.weight"):
                FrozenMTPCapture.from_pretrained(
                    root, device="cpu", dtype=torch.float32
                )


if __name__ == "__main__":
    unittest.main()
