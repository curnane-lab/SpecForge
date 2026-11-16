"""Position identity, compact loss consumption and accelerator parity."""

import pytest
import torch
from torch import nn

from specforge.algorithms.objectives.predictions import Teacher
from specforge.core.fused_loss import ce_l1_loss
from specforge.core.indexed_loss import indexed_ce_l1_loss, indexed_l1_loss
from specforge.core.teacher_projection import (
    IndexedTeacherLogits,
    project_teacher_positions,
    teacher_position_keys,
)
from tests.test_utils.test_fused_loss import DEVICE, _triton_available

DEVICES = ["cpu"] + ([DEVICE] if DEVICE is not None else [])


def test_position_identity_and_chunk_selection():
    # Identical feature values must not merge distinct positions or samples.
    source = torch.ones(2, 4, 9).transpose(1, 2).requires_grad_()
    positions = torch.tensor([[[1, 2, 3], [2, 3, 4]], [[1, 2, 3], [2, 3, 4]]])
    keys = teacher_position_keys(source, positions)
    head = nn.Linear(4, 11)
    rows = []
    head.register_forward_pre_hook(lambda _module, args: rows.append(args[0].shape[0]))
    teacher = Teacher(keys.reshape(4, 3), head=head, source=source)
    first = teacher.select(slice(0, 2)).logits()
    second = teacher.select(slice(2, 4)).logits()
    assert rows == [4, 4]
    for result, batch in ((first, 0), (second, 1)):
        expected = head(source[batch, positions[batch]])
        torch.testing.assert_close(result.values[result.rows], expected)
        assert not result.values.requires_grad
    assert source.grad is None and head.weight.grad is None


@pytest.mark.parametrize("device", DEVICES)
def test_clamping_and_preprojected_source(device):
    source = torch.randn(2, 5, 7, device=device)
    positions = torch.tensor([[[-3, 0, 4, 8]], [[0, 0, 1, 1]]], device=device)
    teacher = project_teacher_positions(
        source, teacher_position_keys(source, positions)
    )
    assert teacher.values.shape == (4, 7)
    batch = torch.arange(2, device=device)[:, None, None]
    torch.testing.assert_close(
        teacher.values[teacher.rows], source[batch, positions.clamp(0, 4)]
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("vocab", [37, 4133, 151936])
def test_ce_l1_values_and_independent_gradients(device, dtype, vocab):
    torch.manual_seed(13)
    logits = torch.randn(2, 3, vocab, device=device, dtype=dtype, requires_grad=True)
    table = torch.randn(4, vocab, device=device, dtype=dtype, requires_grad=True)
    rows = torch.tensor([[0, 1, 1], [2, 3, 2]], device=device)
    labels = torch.randint(vocab, (2, 3), device=device)
    weights = torch.tensor([[0.0, 0.2, 0.8], [1.0, 0.3, 0.0]], device=device)
    expected = ce_l1_loss(logits, labels, teacher_logits=table[rows], top_k=3)

    def objective(terms):
        return (
            (0.1 * terms.cross_entropy + 0.9 * terms.teacher_l1) * weights
        ).sum() + 0.02 * terms.topk_values.sum()

    expected_grad = torch.autograd.grad(objective(expected), logits)[0]
    for fused in (False, True):
        actual = indexed_ce_l1_loss(
            logits,
            labels,
            teacher_logits=IndexedTeacherLogits(table, rows),
            top_k=3,
            fused=fused,
        )
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=3e-5)
        grad, teacher_grad = torch.autograd.grad(
            objective(actual), (logits, table), allow_unused=True
        )
        torch.testing.assert_close(grad, expected_grad, rtol=2e-2, atol=2e-4)
        assert teacher_grad is None


@pytest.mark.parametrize("device", DEVICES)
def test_equal_probabilities_and_saturated_rows(device):
    table = torch.tensor(
        [[0.0, 0.0, 0.0], [100000.0, 99990.0, -100000.0]], device=device
    )
    rows = torch.tensor([[0, 0, 1]], device=device)
    for fused in (False, True):
        logits = table[rows].clone().requires_grad_()
        loss = indexed_l1_loss(logits, IndexedTeacherLogits(table, rows), fused=fused)
        torch.testing.assert_close(loss, torch.zeros_like(loss), rtol=0, atol=0)
        grad = torch.autograd.grad(loss.sum(), logits)[0]
        torch.testing.assert_close(grad, torch.zeros_like(grad), rtol=0, atol=0)


def test_reference_saves_compact_teacher_and_no_expanded_table():
    logits = torch.randn(8, 4133, requires_grad=True)
    table = torch.randn(3, 4133, requires_grad=True)
    rows = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1])
    saved = []

    def pack(value):
        saved.append(value)
        return value

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
        indexed_l1_loss(
            logits, IndexedTeacherLogits(table, rows), fused=False
        ).sum().backward()
    for value in saved:
        if value.shape == logits.shape:
            assert value.data_ptr() == logits.data_ptr()
    assert any(value.shape == table.shape for value in saved)
    assert table.grad is None


@pytest.mark.parametrize("device", DEVICES)
def test_empty_slots_and_invalid_alignment(device):
    logits = torch.empty(0, 3, device=device, requires_grad=True)
    teacher = IndexedTeacherLogits(
        torch.empty(0, 3, device=device),
        torch.empty(0, dtype=torch.long, device=device),
    )
    indexed_l1_loss(logits, teacher).sum().backward()
    assert logits.grad.shape == logits.shape
    with pytest.raises(ValueError, match="match draft slots"):
        indexed_l1_loss(torch.empty(1, 3, device=device), teacher)


@pytest.mark.skipif(
    DEVICE is None or not _triton_available(), reason="accelerator + Triton required"
)
def test_accelerator_really_selects_indexed_kernel():
    from unittest.mock import patch

    from specforge.core.indexed_loss import _IndexedL1Reference

    logits = torch.randn(7, 4099, device=DEVICE, requires_grad=True)
    teacher = IndexedTeacherLogits(
        torch.randn(3, 4099, device=DEVICE),
        torch.tensor([0, 1, 0, 2, 1, 2, 0], device=DEVICE),
    )
    with patch.object(
        _IndexedL1Reference, "forward", side_effect=AssertionError("fallback")
    ):
        indexed_l1_loss(logits, teacher, fused=True).sum().backward()
