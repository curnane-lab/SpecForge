"""DSpark losses consuming a compact teacher table directly.

The accelerator path reuses the portable streaming loss kernels (including
their Ascend block-size cap). The torch fallback tiles the vocabulary, so it
also avoids a [proposal slots, vocabulary] teacher tensor. Teacher gradients
are deliberately absent; each student slot keeps its own loss and gradient.
"""

import torch
import torch.nn.functional as F

from . import fused_loss
from .teacher_projection import IndexedTeacherLogits

_VOCAB_TILE = 4096


def supports_indexed_loss(logits):
    if logits.device.type not in ("cuda", "npu") or logits.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        return False
    try:
        import triton
        import triton.language  # noqa: F401
    except ImportError:
        return False
    return hasattr(triton, "jit")


def _row_stats(values):
    """FP32 online softmax stats, without an FP32 copy of the whole table."""
    maximum = values.new_full((values.shape[0],), -torch.inf, dtype=torch.float32)
    denominator = torch.zeros_like(maximum)
    for start in range(0, values.shape[-1], _VOCAB_TILE):
        tile = values[:, start : start + _VOCAB_TILE].float()
        updated = torch.maximum(maximum, tile.amax(-1))
        denominator = denominator * (maximum - updated).exp() + (
            tile - updated[:, None]
        ).exp().sum(-1)
        maximum = updated
    return maximum, denominator


class _IndexedL1Reference(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, teacher, rows):
        maximum, denominator = _row_stats(logits)
        teacher_max, teacher_den = _row_stats(teacher)
        teacher_max, teacher_den = teacher_max[rows], teacher_den[rows]
        loss = torch.zeros_like(maximum)
        dot = torch.zeros_like(maximum)
        for start in range(0, logits.shape[-1], _VOCAB_TILE):
            p = (
                logits[:, start : start + _VOCAB_TILE].float() - maximum[:, None]
            ).exp() / denominator[:, None]
            q = (
                teacher[:, start : start + _VOCAB_TILE][rows].float()
                - teacher_max[:, None]
            ).exp() / teacher_den[:, None]
            difference = p - q
            loss += difference.abs().sum(-1)
            dot += (p * difference.sign()).sum(-1)
        ctx.save_for_backward(
            logits, teacher, rows, maximum, denominator, teacher_max, teacher_den, dot
        )
        top1, teacher_top1 = denominator.reciprocal(), teacher_den.reciprocal()
        ctx.mark_non_differentiable(top1, teacher_top1)
        return loss, top1, teacher_top1

    @staticmethod
    def backward(ctx, grad_loss, _grad_top1, _grad_teacher_top1):
        logits, teacher, rows, maximum, denominator, teacher_max, teacher_den, dot = (
            ctx.saved_tensors
        )
        grad = torch.empty_like(logits)
        for start in range(0, logits.shape[-1], _VOCAB_TILE):
            p = (
                logits[:, start : start + _VOCAB_TILE].float() - maximum[:, None]
            ).exp() / denominator[:, None]
            q = (
                teacher[:, start : start + _VOCAB_TILE][rows].float()
                - teacher_max[:, None]
            ).exp() / teacher_den[:, None]
            grad[:, start : start + _VOCAB_TILE] = (
                p * ((p - q).sign() - dot[:, None]) * grad_loss[:, None]
            )
        return grad, None, None


class _IndexedL1Fused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, teacher, rows):
        fused_loss._require_triton()
        count, vocab = logits.shape
        loss = torch.empty(count, device=logits.device, dtype=torch.float32)
        stats = torch.empty(5, count, device=logits.device, dtype=torch.float32)
        block, warps = fused_loss._calculate_settings(vocab, logits.device)
        if count:
            fused_loss.loss_forward_kernel[(count,)](
                logits,
                teacher,
                loss,
                stats,
                count,
                vocab,
                OP=fused_loss._OP_L1.value,
                BLOCK_SIZE=block,
                target_rows_ptr=rows,
                INDEXED_TARGETS=True,
                num_warps=warps,
            )
        ctx.save_for_backward(logits, teacher, rows, stats)
        ctx.settings = block, warps
        top1, teacher_top1 = stats[1].reciprocal(), stats[3].reciprocal()
        ctx.mark_non_differentiable(top1, teacher_top1)
        return loss, top1, teacher_top1

    @staticmethod
    def backward(ctx, grad_loss, _grad_top1, _grad_teacher_top1):
        logits, teacher, rows, stats = ctx.saved_tensors
        count, vocab = logits.shape
        block, warps = ctx.settings
        grad = torch.empty_like(logits)
        if count:
            fused_loss.loss_backward_kernel[(count,)](
                logits,
                teacher,
                grad,
                grad_loss.contiguous(),
                stats,
                count,
                vocab,
                OP=fused_loss._OP_L1.value,
                BLOCK_SIZE=block,
                target_rows_ptr=rows,
                INDEXED_TARGETS=True,
                num_warps=warps,
            )
        return grad, None, None


def _indexed_l1(logits, teacher, *, fused):
    if (
        logits.ndim < 2
        or logits.shape[-1] == 0
        or teacher.rows.shape != logits.shape[:-1]
        or teacher.values.shape[-1] != logits.shape[-1]
        or teacher.values.device != logits.device
    ):
        raise ValueError(
            "indexed teacher must match draft slots, vocabulary and device"
        )
    implementation = (
        _IndexedL1Fused
        if fused and supports_indexed_loss(logits)
        else _IndexedL1Reference
    )
    result = implementation.apply(
        logits.reshape(-1, logits.shape[-1]).contiguous(),
        teacher.values.detach().contiguous(),
        teacher.rows.flatten().contiguous(),
    )
    return tuple(value.reshape(logits.shape[:-1]) for value in result)


def indexed_l1_loss(logits, teacher: IndexedTeacherLogits, *, fused=True):
    """Per-slot L1; vocabulary tiles are temporary, the teacher stays unique."""
    return _indexed_l1(logits, teacher, fused=fused)[0]


def indexed_ce_l1_loss(
    logits, target_ids, *, teacher_logits: IndexedTeacherLogits, top_k=0, fused=True
):
    """Explicit-label CE plus indexed teacher L1, with the CEL1Loss contract.

    CE is unchanged. Only L1 needs a row-indexed backend. Promote draft logits
    once so CE/L1/top-k gradients combine before casting back to BF16/FP16.
    """
    if target_ids.shape != logits.shape[:-1] or not 0 <= top_k <= logits.shape[-1]:
        raise ValueError("target_ids shape or top_k does not match logits")
    promoted = logits.to(torch.promote_types(logits.dtype, torch.float32))
    ce = F.cross_entropy(
        promoted.reshape(-1, logits.shape[-1]), target_ids.flatten(), reduction="none"
    ).reshape_as(target_ids)
    l1, top1, teacher_top1 = _indexed_l1(promoted, teacher_logits, fused=fused)
    topv, topi = promoted.topk(top_k, dim=-1)
    with torch.no_grad():
        teacher_ids = teacher_logits.values.argmax(-1)[teacher_logits.rows]
    return fused_loss.CEL1Loss(
        cross_entropy=ce,
        teacher_l1=l1,
        topk_values=topv,
        topk_ids=topi,
        argmax_ids=logits.argmax(-1),
        top1_probability=top1,
        teacher_argmax_ids=teacher_ids,
        teacher_top1_probability=teacher_top1,
    )
