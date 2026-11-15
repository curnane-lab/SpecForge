"""Train through legacy and canonical DFlash entry points with bounded attention."""

import copy
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel
from specforge.algorithms.objectives import build_objective_model
from specforge.algorithms.objectives.inputs import prepare_blocks
from specforge.application.composition import resolve_run
from specforge.config import Config, TrainingConfig
from specforge.modeling import build_draft_model
from specforge.modeling.draft.anchor_attention import AnchorBlockMask
from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.training.assembly import build_model_bundle
from tests.test_algorithms.test_objectives import training_batch
from tests.test_config.draft_fixtures import draft_config
from tests.test_modeling.legacy_fixtures import legacy_dflash_config
from tests.test_modeling.test_canonical_dflash2 import dflash2_config
from tests.test_runtime.test_canonical_training import run_config, target_directory


def assert_parameter_gradients(test, actual, expected):
    expected_parameters = dict(expected.named_parameters())
    for name, parameter in actual.named_parameters():
        reference = expected_parameters[name]
        test.assertEqual(parameter.grad is None, reference.grad is None, name)
        if parameter.grad is not None:
            torch.testing.assert_close(
                parameter.grad, reference.grad, rtol=5e-4, atol=2e-5, msg=name
            )


class AnchorTrainingTest(unittest.TestCase):
    def test_canonical_dflash_and_dflash2_loss_and_all_parameter_gradients(self):
        for factory in (
            lambda: draft_config(target_bindings=(), target_logit_transform=None),
            dflash2_config,
        ):
            for layer_checkpointing in (False, True):
                with self.subTest(
                    factory=factory, layer_checkpointing=layer_checkpointing
                ):
                    torch.manual_seed(17)
                    config = factory()
                    reference = build_draft_model(config, backend="sdpa")
                    actual = copy.deepcopy(reference)
                    if layer_checkpointing:
                        reference.gradient_checkpointing_enable(
                            gradient_checkpointing_kwargs={"use_reentrant": False}
                        )
                        actual.gradient_checkpointing_enable(
                            gradient_checkpointing_kwargs={"use_reentrant": False}
                        )
                    options = {"num_anchors": 5}
                    runtimes = [
                        build_objective_model(reference, options=options),
                        build_objective_model(
                            actual,
                            options={
                                **options,
                                "attention_chunk_blocks": 2,
                                "attention_trim_context": True,
                            },
                        ),
                    ]
                    batch = training_batch(config)
                    # The second sample has no valid anchor; it still participates in backward.
                    batch = {
                        key: value.detach().repeat(2, *([1] * (value.ndim - 1)))
                        for key, value in batch.items()
                    }
                    batch["loss_mask"][1] = 0
                    losses = []
                    for runtime in runtimes:
                        torch.manual_seed(31)
                        result = runtime(batch, step=1, total_steps=4)
                        result.loss.backward()
                        losses.append(result.loss)
                    torch.testing.assert_close(
                        losses[0], losses[1], rtol=2e-5, atol=2e-6
                    )
                    assert_parameter_gradients(self, actual, reference)

    def test_legacy_wrapper_loss_features_and_all_parameter_gradients(self):
        for causal, window in ((False, None), (True, 4), (False, 4)):
            with self.subTest(causal=causal, window=window):
                torch.manual_seed(23)
                config = legacy_dflash_config(
                    block_size=4,
                    is_causal=causal,
                    layer_types=(
                        ["sliding_attention", "full_attention"]
                        if window
                        else ["full_attention"] * 2
                    ),
                    sliding_window=window,
                    use_sliding_window=window is not None,
                    dflash_config={"target_layer_ids": [1, 3], "mask_token_id": 63},
                )
                config._attn_implementation = "sdpa"
                reference = OnlineDFlashModel(
                    DFlashDraftModel(config),
                    nn.Linear(32, 64, bias=False),
                    nn.Embedding(64, 32),
                    mask_token_id=63,
                    block_size=4,
                    num_anchors=5,
                    attention_backend="sdpa",
                    fused_logits_loss=False,
                    objective_checkpointing=False,
                )
                actual = copy.deepcopy(reference)
                actual.attention_chunk_blocks = 2
                actual.attention_trim_context = True
                features = torch.randn(2, 13, 64)
                reference_features = features.clone().requires_grad_()
                actual_features = features.clone().requires_grad_()
                ids = torch.randint(0, 60, (2, 13))
                loss_mask = torch.tensor(
                    [[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], [0] * 13]
                )
                losses = []
                for model, hidden in (
                    (reference, reference_features),
                    (actual, actual_features),
                ):
                    torch.manual_seed(31)
                    loss, _, _ = model(
                        ids, hidden, loss_mask, collect_detailed_metrics=False
                    )
                    loss.backward()
                    losses.append(loss)
                torch.testing.assert_close(losses[0], losses[1], rtol=2e-5, atol=2e-6)
                torch.testing.assert_close(
                    actual_features.grad, reference_features.grad, rtol=5e-4, atol=2e-5
                )
                assert_parameter_gradients(self, actual, reference)

    def test_prepare_blocks_retains_sampling_and_uses_compact_visibility(self):
        config = draft_config()
        batch = training_batch(config)
        torch.manual_seed(5)
        expected = prepare_blocks(config, batch, num_anchors=5)
        torch.manual_seed(5)
        actual = prepare_blocks(config, batch, num_anchors=5, attention_chunk_blocks=2)
        for name in (
            "input_ids",
            "labels",
            "predecessors",
            "weights",
            "anchors",
            "keep",
            "label_positions",
        ):
            torch.testing.assert_close(
                getattr(actual, name), getattr(expected, name), rtol=0, atol=0
            )
        self.assertIsInstance(
            actual.layer_inputs[0].attention_layout.visibility, AnchorBlockMask
        )

    @patch("specforge.utils.get_local_device", return_value=torch.device("cpu"))
    def test_legacy_config_build_passes_tile_size_to_the_training_model(self, _device):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = target_directory(root)
            draft = legacy_dflash_config(
                architectures=["DFlashDraftModel"],
                block_size=4,
                dflash_config={"target_layer_ids": [1, 3], "mask_token_id": 63},
            )
            draft_path = root / "draft.json"
            draft.to_json_file(draft_path)
            cfg = Config.model_validate(
                dict(
                    model={
                        "target_model_path": str(target),
                        "draft_model_config": str(draft_path),
                        "torch_dtype": "float32",
                    },
                    data={"hidden_states_path": str(root / "features")},
                    training={
                        "strategy": "dflash",
                        "attention_backend": "sdpa",
                        "attention_chunk_blocks": 2,
                        "attention_trim_context": True,
                    },
                )
            )
            resolved = resolve_run(cfg)
            with patch(
                "specforge.training.assembly._load_input_tools",
                return_value=SimpleNamespace(),
            ):
                bundle = build_model_bundle(
                    resolved.config, algorithm=resolved.algorithm
                )
            self.assertEqual(bundle.model.attention_chunk_blocks, 2)
            self.assertEqual(next(bundle.draft_model.parameters()).device.type, "cpu")
            self.assertEqual(
                bundle.strategy_kwargs.resume_contract["dflash_attention_backend"],
                {
                    "backend": "sdpa",
                    "attention_chunk_blocks": 2,
                    "attention_trim_context": True,
                },
            )
            self.assertIsInstance(
                bundle.model._attention_mask(
                    torch.tensor([[1, 4]]), torch.tensor([[True, True]]), 9
                ),
                AnchorBlockMask,
            )

    @patch("specforge.utils.get_local_device", return_value=torch.device("cpu"))
    def test_normal_config_resolution_build_and_resume_contract(self, _device):
        with TemporaryDirectory() as tmp:
            target = target_directory(Path(tmp))
            for chunk in (0, 2):
                cfg = run_config(
                    target,
                    draft_config(),
                    training={
                        "attention_backend": "sdpa",
                        "attention_chunk_blocks": chunk,
                        "attention_trim_context": bool(chunk),
                    },
                )
                resolved = resolve_run(cfg)
                bundle = build_model_bundle(
                    resolved.config, algorithm=resolved.algorithm
                )
                self.assertEqual(bundle.model.objective.attention_chunk_blocks, chunk)
                self.assertEqual(
                    bundle.model.objective.attention_trim_context, bool(chunk)
                )
                self.assertEqual(
                    next(bundle.draft_model.parameters()).device.type, "cpu"
                )
                options = bundle.strategy_kwargs.resume_contract[
                    "canonical_objective_options"
                ]
                self.assertEqual(options.get("attention_chunk_blocks", 0), chunk)
                self.assertEqual("attention_chunk_blocks" in options, bool(chunk))
                self.assertEqual("attention_trim_context" in options, bool(chunk))
                result = bundle.model(
                    training_batch(bundle.draft_model.config.draft),
                    step=1,
                    total_steps=2,
                )
                result.loss.backward()
                self.assertTrue(torch.isfinite(result.loss))

    def test_configuration_rejects_unsupported_combinations(self):
        for value in (-1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                TrainingConfig(attention_backend="sdpa", attention_chunk_blocks=value)
        with self.assertRaisesRegex(ValueError, "attention_backend=sdpa"):
            TrainingConfig(attention_backend="eager", attention_chunk_blocks=2)
        with self.assertRaisesRegex(ValueError, "requires attention_chunk_blocks"):
            TrainingConfig(attention_backend="sdpa", attention_trim_context=True)
        with self.assertRaisesRegex(ValueError, "attention_trim_context"):
            TrainingConfig(
                attention_backend="sdpa",
                attention_chunk_blocks=2,
                attention_trim_context=1,
            )
        with TemporaryDirectory() as tmp:
            target = target_directory(Path(tmp))
            cfg = run_config(
                target,
                draft_config(),
                training={
                    "attention_backend": "sdpa",
                    "attention_chunk_blocks": 2,
                    "objective_options": {"attention_chunk_blocks": 3},
                },
            )
            with self.assertRaisesRegex(
                ValueError, "conflicting attention_chunk_blocks"
            ):
                resolve_run(cfg)
            from tests.test_modeling.test_mtp_sdpa_chunking import owned_mtp_config

            cfg = run_config(
                target,
                owned_mtp_config(),
                training={"attention_backend": "sdpa", "attention_chunk_blocks": 2},
            )
            with self.assertRaisesRegex(ValueError, "DFlash"):
                resolve_run(cfg)
