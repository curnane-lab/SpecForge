"""DSpark training entry points, checkpoint replay and projection row counts."""

import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest
import torch
from torch import nn

from specforge.algorithms.objectives import build_objective_model, resolve_objective
from specforge.application.composition import resolve_run
from specforge.config import Config
from specforge.core.teacher_projection import teacher_position_keys
from specforge.modeling import build_draft_model
from specforge.training.assembly import build_model_bundle
from tests.test_algorithms.test_objective_legacy_parity import block_pair
from tests.test_algorithms.test_objectives import training_batch
from tests.test_modeling.test_canonical_dspark import model_config
from tests.test_runtime.test_canonical_training import run_config, target_directory
from tests.test_utils.test_fused_loss import DEVICE


def assert_parameter_gradients(left, right):
    other = dict(right.named_parameters())
    for name, parameter in left.named_parameters():
        expected = other[name].grad
        assert (parameter.grad is None) == (expected is None), name
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name
            torch.testing.assert_close(
                parameter.grad, expected, rtol=3e-4, atol=3e-6, msg=name
            )


@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("chunk", [0, 1, 2])
def test_legacy_full_forward_backward(checkpoint, chunk):
    torch.manual_seed(7)
    _, dense, _ = block_pair("dspark")
    dense = dense.to("cpu")
    compact = copy.deepcopy(dense)
    compact.deduplicate_teacher_positions = True
    for model in (dense, compact):
        model.objective_chunk_blocks = chunk
        model.objective_checkpointing = checkpoint
    batch = dict(
        input_ids=torch.randint(0, 62, (2, 12)),
        hidden_states=torch.randn(2, 12, 64),
        target_last_hidden_states=torch.randn(2, 12, 32, requires_grad=True),
        loss_mask=torch.ones(2, 12),
    )
    anchors = torch.tensor([[2, 3, 4], [0, 0, 0]])
    keep = torch.tensor([[True, True, True], [False, False, False]])
    batch["loss_mask"][1] = 0
    projected_rows = []

    def count_teacher_rows(_head, args):
        if not torch.is_grad_enabled():
            projected_rows.append(args[0].numel() // args[0].shape[-1])

    compact.lm_head.register_forward_pre_hook(count_teacher_rows)
    losses = []
    for model in (dense, compact):
        with patch.object(
            model, "_sample_anchor_positions", return_value=(anchors, keep)
        ):
            losses.append(model(**batch))
    torch.testing.assert_close(losses[0][:2], losses[1][:2], rtol=2e-5, atol=2e-6)
    left, right = losses[0][2]["ratio_metrics"], losses[1][2]["ratio_metrics"]
    assert left.keys() == right.keys()
    for name in left:
        torch.testing.assert_close(left[name], right[name], rtol=2e-5, atol=3e-6)

    _, _, labels = compact._build_dspark_labels_and_mask(
        batch["input_ids"], batch["loss_mask"], anchors, keep
    )
    keys = teacher_position_keys(batch["target_last_hidden_states"], labels - 1)
    expected_rows = [
        keys[:, start : start + (chunk or 3)].unique().numel()
        for start in range(0, 3, chunk or 3)
    ]
    assert projected_rows == expected_rows
    losses[0][0].backward()
    losses[1][0].backward()
    assert projected_rows == expected_rows + (
        list(reversed(expected_rows)) if checkpoint else []
    )
    assert sum(expected_rows) < 2 * 3 * 4
    assert_parameter_gradients(compact, dense)
    assert batch["target_last_hidden_states"].grad is None


@pytest.mark.parametrize("chunk", [0, 2])
@pytest.mark.parametrize("preprojected", [False, True])
def test_canonical_full_forward_backward(chunk, preprojected):
    torch.manual_seed(15)
    config = model_config()
    dense = build_objective_model(
        build_draft_model(config, device="cpu"),
        target_head=nn.Linear(32, config.vocab_size, bias=False),
        options={"num_anchors": 3, "objective_chunk_size": chunk},
    )
    compact = copy.deepcopy(dense)
    compact.objective.deduplicate_teacher_positions = True
    batch = training_batch(config)
    batch = {
        key: torch.cat([value, value.detach()], dim=0)
        .detach()
        .requires_grad_(value.requires_grad)
        for key, value in batch.items()
    }
    batch["loss_mask"][1] = 0  # One sample has no valid anchors.
    if preprojected:
        batch["target_logits"] = (
            dense.target_head(batch["target_last_hidden_states"])
            .detach()
            .requires_grad_()
        )
    results = []
    for model in (dense, compact):
        torch.manual_seed(23)
        results.append(model(batch, step=1, total_steps=10))
    torch.testing.assert_close(results[0].loss, results[1].loss, rtol=2e-5, atol=2e-6)
    for result in results:
        result.loss.backward()
    assert_parameter_gradients(compact, dense)
    assert all(value.grad is None for value in batch.values() if value.requires_grad)


def test_teacher_not_projected_when_legacy_l1_and_confidence_disabled():
    _, model, _ = block_pair("dspark")
    model = model.to("cpu")
    model.deduplicate_teacher_positions = True
    model.dspark_l1_loss_alpha = model.dspark_confidence_head_alpha = 0
    tokens = torch.randint(0, 62, (1, 12))
    teacher_shapes = []
    model.lm_head.register_forward_pre_hook(
        lambda _head, args: teacher_shapes.append(torch.is_grad_enabled())
    )
    model(tokens, torch.randn(1, 12, 64), torch.ones_like(tokens))[0].backward()
    assert all(teacher_shapes)


def test_config_wiring_and_resume_contract_unchanged():
    with TemporaryDirectory() as tmp:
        target = target_directory(Path(tmp))
        cfg = run_config(target, model_config())
        contracts = []
        for enabled in (False, True):
            raw = cfg.model_dump()
            raw["training"]["deduplicate_teacher_positions"] = enabled
            resolved = resolve_run(Config.model_validate(raw))
            bundle = build_model_bundle(resolved.config, algorithm=resolved.algorithm)
            assert bundle.model.objective.deduplicate_teacher_positions is enabled
            contracts.append(bundle.strategy_kwargs.resume_contract)
        assert contracts[0] == contracts[1]
        assert (
            "deduplicate_teacher_positions"
            not in contracts[0]["canonical_objective_options"]
        )


def test_incompatible_loss_and_non_dspark_settings_fail_early():
    with pytest.raises(ValueError, match="supports ce and ce_l1"):
        resolve_objective(
            "DSparkDraftModel",
            options={"deduplicate_teacher_positions": True},
            loss={"kind": "kl"},
        )
    with pytest.raises(ValueError, match="must be a bool"):
        resolve_objective(
            "DSparkDraftModel", options={"deduplicate_teacher_positions": "true"}
        )
    with TemporaryDirectory() as tmp:
        from tests.test_config.draft_fixtures import draft_config

        target = target_directory(Path(tmp))
        cfg = run_config(
            target,
            draft_config(),
            training={
                "attention_backend": "eager",
                "deduplicate_teacher_positions": True,
            },
        )
        with pytest.raises(ValueError, match="requires DSpark"):
            resolve_run(cfg)
    legacy = Config.model_validate(
        {
            "model": {"target_model_path": "unused", "draft_model_config": "unused"},
            "data": {"hidden_states_path": "unused"},
            "training": {
                "strategy": "dflash",
                "attention_backend": "sdpa",
                "deduplicate_teacher_positions": True,
            },
        }
    )
    with pytest.raises(ValueError, match="requires DSpark"):
        resolve_run(legacy)


@pytest.mark.skipif(DEVICE is None, reason="NPU or CUDA required")
@pytest.mark.parametrize("canonical", [False, True])
def test_accelerator_bf16_model_forward_backward(canonical):
    torch.manual_seed(29)
    if canonical:
        config = model_config()
        dense = build_objective_model(
            build_draft_model(config, device=DEVICE, dtype=torch.bfloat16),
            target_head=nn.Linear(
                32, config.vocab_size, bias=False, device=DEVICE, dtype=torch.bfloat16
            ),
            options={"num_anchors": 3, "objective_chunk_size": 2},
        )
        batch = training_batch(config, device=DEVICE)
        compact = copy.deepcopy(dense)
        compact.objective.deduplicate_teacher_positions = True

        def forward(model):
            return model(batch, step=1, total_steps=10).loss

    else:
        _, dense, _ = block_pair("dspark")
        dense = dense.to(device=DEVICE, dtype=torch.bfloat16)
        dense.objective_chunk_blocks = 2
        compact = copy.deepcopy(dense)
        compact.deduplicate_teacher_positions = True
        tokens = torch.randint(0, 62, (1, 12), device=DEVICE)
        batch = dict(
            input_ids=tokens,
            hidden_states=torch.randn(1, 12, 64, device=DEVICE, dtype=torch.bfloat16),
            target_last_hidden_states=torch.randn(
                1, 12, 32, device=DEVICE, dtype=torch.bfloat16
            ),
            loss_mask=torch.ones_like(tokens),
        )

        def forward(model):
            return model(**batch)[0]

    losses = []
    for model in (dense, compact):
        torch.manual_seed(31)
        loss = forward(model)
        loss.backward()
        losses.append(loss.detach())
    torch.testing.assert_close(losses[0], losses[1], rtol=2e-3, atol=3e-3)
    expected = dict(dense.named_parameters())
    for name, parameter in compact.named_parameters():
        if expected[name].grad is None:
            assert parameter.grad is None
        else:
            assert (
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
            ), name
            torch.testing.assert_close(
                parameter.grad, expected[name].grad, rtol=3e-2, atol=3e-3, msg=name
            )
