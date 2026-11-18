import unittest

import torch
import yaml

from specforge.algorithms.objectives import build_objective_model
from specforge.config.draft.codec import decode_config
from specforge.modeling import build_draft_model
from tests.test_algorithms.test_objectives import training_batch
from tests.test_modeling.test_canonical_dmtp import model_config
from tests.test_utils.test_dmtp_capture import npu_available


class DMTPNPUTest(unittest.TestCase):
    @unittest.skipUnless(npu_available(), "DMTP NPU test requires Ascend NPU")
    def test_objective_forward_backward(self):
        config = model_config()
        model = build_draft_model(config, backend="sdpa", device="npu")
        runtime = build_objective_model(model, options={"num_anchors": 2})
        batch = training_batch(config, "npu")
        result = runtime(batch, step=1, total_steps=10)
        self.assertTrue(torch.isfinite(result.loss))
        result.loss.backward()
        self.assertGreater(model.mtp_fusion.fc.weight.grad.abs().sum().item(), 0)
        self.assertEqual(
            [
                name
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
                and (parameter.grad is None or not torch.isfinite(parameter.grad).all())
            ],
            [],
        )
        self.assertTrue(
            all(
                tensor.grad is None for tensor in batch.values() if tensor.requires_grad
            )
        )

    def test_qwen36_canonical_preset(self):
        with open("specforge/config/draft/presets/qwen3.6-27b-dmtp.yaml") as stream:
            payload = yaml.safe_load(stream)
        config = decode_config(
            payload,
            target_config={
                "hidden_size": 5120,
                "vocab_size": 248320,
                "num_hidden_layers": 64,
            },
        )
        self.assertEqual(config.architecture, "DMTPDraftModel")
        self.assertEqual(config.spec_config.protocol.num_model_slots, 7)
        self.assertEqual(config.spec_config.protocol.num_draft_tokens, 6)
        self.assertEqual(config.spec_config.context_end_offset, -1)
        self.assertEqual(config.spec_config.mtp_features.hidden_size, 5120)


if __name__ == "__main__":
    unittest.main()
