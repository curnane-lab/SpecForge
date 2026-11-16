"""Project frozen teacher positions once, without expanding vocabulary rows."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class IndexedTeacherLogits:
    """A [unique positions, vocab] table and one table index per draft slot.

    ``rows`` must contain valid indices into ``values``. Keeping the inverse map
    separate is essential: indexing the full table by ``rows`` would recreate
    the duplicate full-vocabulary allocation that this representation avoids.
    Neither the table nor the map is differentiable.
    """

    values: Tensor
    rows: Tensor

    def __post_init__(self):
        if self.values.ndim != 2 or not self.values.is_floating_point():
            raise ValueError("teacher values must be a floating [unique, vocab] table")
        if self.rows.dtype != torch.long or self.rows.device != self.values.device:
            raise ValueError("teacher rows must be int64 on the teacher device")

    @property
    def shape(self):
        """Logical slot/vocabulary shape for loss alignment checks."""
        return torch.Size((*self.rows.shape, self.values.shape[-1]))


def teacher_position_keys(source: Tensor, positions: Tensor) -> Tensor:
    """Encode (sample, clamped position), preserving the existing gather policy."""
    if source.ndim != 3 or source.shape[1] == 0:
        raise ValueError("teacher source must be [batch, nonempty sequence, hidden]")
    if (
        positions.ndim < 2
        or positions.shape[0] != source.shape[0]
        or positions.dtype != torch.long
        or positions.device != source.device
    ):
        raise ValueError("teacher positions must be aligned int64 batch indices")
    offset = (
        torch.arange(source.shape[0], device=source.device).reshape(
            -1, *([1] * (positions.ndim - 1))
        )
        * source.shape[1]
    )
    return positions.clamp(0, source.shape[1] - 1) + offset


@torch.no_grad()
def project_teacher_positions(source: Tensor, keys: Tensor, head=None):
    """Deduplicate one objective chunk; never cache across forwards or ranks.

    Gather before projecting, including for noncontiguous capture tensors. The
    sequence itself is not flattened/copied. ``head=None`` accepts already
    projected teacher logits and compacts their rows using the same protocol.
    """
    unique, inverse = torch.unique(keys.flatten(), sorted=True, return_inverse=True)
    selected = source[unique // source.shape[1], unique % source.shape[1]]
    values = selected if head is None else head(selected)
    return IndexedTeacherLogits(values.detach(), inverse.reshape_as(keys))
