# coding=utf-8
"""Target-aware prefix compression modules for DFlash.

This module provides small learnable compressors that reduce the target model's
input sequence length during prefill (e.g. by 2x).  The compressed embeddings are
fed to the target model in place of the original token embeddings, speeding up
TTFT and reducing KV-cache memory.  A hidden-state preservation loss can be used
to train the compressor so that the compressed prefix remains semantically
useful for the DFlash draft model.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Use PyTorch native RMSNorm when available (2.6+), otherwise provide a tiny
# fallback implementation that matches the standard RMSNorm semantics.
try:
    RMSNorm = nn.RMSNorm
except AttributeError:

    class RMSNorm(nn.Module):
        """Simple RMSNorm fallback."""

        def __init__(self, hidden_size: int, eps: float = 1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.eps = eps

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
            return self.weight * hidden_states.to(input_dtype)


def _reshape_to_3d(x: torch.Tensor) -> torch.Tensor:
    """Promote a 2D [T, H] tensor to 3D [1, T, H]; leave 3D inputs unchanged."""
    if x.dim() == 2:
        return x.unsqueeze(0)
    if x.dim() != 3:
        raise ValueError(f"Expected 2D or 3D input, got {x.dim()}D")
    return x


def _restore_from_3d(x: torch.Tensor, original_dim: int) -> torch.Tensor:
    """Undo _reshape_to_3d when the original input was 2D."""
    if original_dim == 2:
        return x.squeeze(0)
    return x


def compress_mask(mask: torch.Tensor, compress_ratio: int) -> torch.Tensor:
    """Compress a binary mask by taking the max over each ``compress_ratio`` group.

    Supports 2D [B, T] and 3D [B, T, ...] masks.  The last non-batch dimension is
    grouped and compressed with ``max`` so that any valid token in a group keeps
    the group valid.
    """
    if mask.dim() not in (2, 3):
        raise ValueError(f"Expected 2D or 3D mask, got {mask.dim()}D")
    bsz, seq_len = mask.shape[:2]
    if seq_len % compress_ratio != 0:
        pad = compress_ratio - (seq_len % compress_ratio)
        pad_shape = (bsz, pad) + mask.shape[2:]
        mask = torch.cat(
            [mask, torch.zeros(pad_shape, dtype=mask.dtype, device=mask.device)], dim=1
        )
        seq_len = mask.shape[1]
    # [B, T//r, r, ...]
    grouped = mask.view(bsz, seq_len // compress_ratio, compress_ratio, *mask.shape[2:])
    return grouped.max(dim=2).values


class BaseCompressor(nn.Module, ABC):
    """Abstract base class for DFlash prefix compressors."""

    @abstractmethod
    def forward(self, input_embeds: torch.Tensor) -> torch.Tensor:
        """Compress ``input_embeds`` along the sequence dimension.

        Args:
            input_embeds: Token embeddings of shape ``[B, T, H]`` or ``[T, H]``.

        Returns:
            Compressed embeddings of shape ``[B, T//r, H]`` or ``[T//r, H]``.
        """


class EmbeddingPairCompressor(BaseCompressor):
    """Compress pairs (or groups) of token embeddings via a Linear or small MLP.

    The module is initialized so that, near the origin, the output approximates
    the sum of the embeddings in each group, i.e.
    ``f(e_i, e_{i+1}, ...) ≈ e_i + e_{i+1} + ...``.  This preserves the
    ``skip-connection'' semantics of the original prefix.
    """

    def __init__(
        self,
        config,
        compress_ratio: int = 2,
        compressor_type: str = "mlp",
    ):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.compressor_type = compressor_type.lower()
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("config must have a 'hidden_size' attribute")
        self.hidden_size = hidden_size

        in_dim = compress_ratio * hidden_size
        if self.compressor_type == "linear":
            self.projector = nn.Linear(in_dim, hidden_size, bias=False)
        elif self.compressor_type == "mlp":
            self.projector = nn.Sequential(
                nn.Linear(in_dim, hidden_size, bias=False),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size, bias=False),
            )
        else:
            raise ValueError(
                f"Unknown compressor_type={compressor_type!r}; "
                "expected 'linear' or 'mlp'"
            )

        self.norm = RMSNorm(hidden_size, eps=getattr(config, "rms_norm_eps", 1e-6))
        self._init_to_sum()

    def _init_to_sum(self) -> None:
        """Initialize the projector so that each output is the sum of its inputs."""
        if self.compressor_type == "linear":
            # W = [I | I | ... | I], b = 0  =>  W @ [e0;e1;...] = sum(e_i)
            weight = torch.zeros(
                self.hidden_size, self.compress_ratio * self.hidden_size
            )
            for i in range(self.compress_ratio):
                weight[:, i * self.hidden_size : (i + 1) * self.hidden_size] = (
                    torch.eye(self.hidden_size)
                )
            with torch.no_grad():
                self.projector.weight.copy_(weight)
        else:
            # MLP: first layer maps [e0;...] -> sum(e_i), second layer is identity.
            # SiLU(0)=0 and SiLU'(0)=1, so near the origin f(x) ≈ x.
            first = self.projector[0]
            second = self.projector[2]
            w1 = torch.zeros(self.hidden_size, self.compress_ratio * self.hidden_size)
            for i in range(self.compress_ratio):
                w1[:, i * self.hidden_size : (i + 1) * self.hidden_size] = torch.eye(
                    self.hidden_size
                )
            with torch.no_grad():
                first.weight.copy_(w1)
                second.weight.copy_(torch.eye(self.hidden_size))

    def forward(self, input_embeds: torch.Tensor) -> torch.Tensor:
        original_dim = input_embeds.dim()
        x = _reshape_to_3d(input_embeds)
        bsz, seq_len, hidden_size = x.shape
        if seq_len % self.compress_ratio != 0:
            pad = self.compress_ratio - (seq_len % self.compress_ratio)
            x = F.pad(x, (0, 0, 0, pad), value=0.0)
            seq_len = x.shape[1]

        # [B, T//r, r*H]
        x = x.view(bsz, seq_len // self.compress_ratio, -1)
        x = self.projector(x)
        x = self.norm(x)
        return _restore_from_3d(x, original_dim)


class EarlySemanticCompressor(BaseCompressor):
    """Run a few early transformer layers before pair/group compression.

    If a ``target_model`` is provided, the early layers are *best-effort*
    initialized from the target model's first ``early_layers`` decoder layers.
    Because target-model layers are architecture-specific (GQA, fused MLPs,
    RoPE, etc.), the copy is approximate; when exact copying is not possible a
    warning is emitted and the layers remain randomly initialized.  If no target
    model is supplied, generic PyTorch ``TransformerEncoderLayer`` blocks are
    created from scratch.
    """

    def __init__(
        self,
        config,
        early_layers: int = 2,
        compress_ratio: int = 2,
        compressor_type: str = "mlp",
        target_model: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.early_layers = early_layers
        self.compress_ratio = compress_ratio
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("config must have a 'hidden_size' attribute")
        self.hidden_size = hidden_size

        num_heads = getattr(config, "num_attention_heads", max(1, hidden_size // 64))
        intermediate_size = getattr(config, "intermediate_size", 4 * hidden_size)
        dropout = getattr(config, "attention_dropout", 0.0)
        eps = getattr(config, "rms_norm_eps", 1e-6)

        # Generic transformer encoder layers with Pre-LN (norm_first=True).
        # Use GELU because PyTorch's TransformerEncoderLayer supports only
        # relu/gelu out of the box.  This is a generic fallback; a model-specific
        # early-semantic implementation could reuse the target architecture.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=intermediate_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.semantic_layers = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=early_layers,
            norm=RMSNorm(hidden_size, eps=eps),
        )

        if target_model is not None:
            self._try_init_from_target(target_model)

        self.pair_compressor = EmbeddingPairCompressor(
            config=config,
            compress_ratio=compress_ratio,
            compressor_type=compressor_type,
        )

    def _try_init_from_target(self, target_model: nn.Module) -> None:
        """Best-effort weight copy from the target model's early layers."""
        source_layers = _find_target_layers(target_model, self.early_layers)
        if source_layers is None:
            import warnings

            warnings.warn(
                "Could not locate decoder layers in target_model for early-semantic "
                "compressor initialization. Layers will remain randomly initialized."
            )
            return

        target_layers = list(self.semantic_layers.encoder.layers)
        for idx, (src, dst) in enumerate(zip(source_layers, target_layers)):
            try:
                _copy_layer_weights(src, dst)
            except Exception as exc:  # noqa: BLE001
                import warnings

                warnings.warn(
                    f"Failed to copy weights for early-semantic layer {idx}: {exc}. "
                    "Layer remains randomly initialized."
                )

    def forward(self, input_embeds: torch.Tensor) -> torch.Tensor:
        original_dim = input_embeds.dim()
        x = _reshape_to_3d(input_embeds)
        # Prefix compression is bidirectional; no causal mask is required.
        x = self.semantic_layers(x)
        x = self.pair_compressor(x)
        return _restore_from_3d(x, original_dim)


def _find_target_layers(target_model: nn.Module, early_layers: int):
    """Try a few common attribute paths for target-model decoder layers."""
    candidates = [
        "model.layers",
        "model.model.layers",
        "transformer.h",
        "model.decoder.layers",
        "decoder.layers",
    ]
    for attr in candidates:
        obj = target_model
        for part in attr.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None and len(obj) >= early_layers:
            return list(obj[:early_layers])
    return None


def _copy_layer_weights(src: nn.Module, dst: nn.Module) -> None:
    """Best-effort copy from a HF-style decoder layer to a generic encoder layer.

    The mapping assumes ``src`` follows the common LLaMA/Qwen layout:
    ``self_attn.{q,k,v,o}_proj``, ``mlp.{gate,up,down}_proj`` and layer norms.
    When shapes do not line up (e.g. GQA key/value heads), the matching slices
    are copied and the remainder is left random.
    """
    src_attn = getattr(src, "self_attn", None)
    dst_attn = getattr(dst, "self_attn", None)
    if src_attn is not None and dst_attn is not None:
        # q_proj -> first H rows of in_proj_weight
        q = getattr(src_attn, "q_proj", None)
        if q is not None:
            h = q.weight.shape[0]
            with torch.no_grad():
                dst_attn.in_proj_weight[:h].copy_(q.weight)
        # k_proj -> middle H rows
        k = getattr(src_attn, "k_proj", None)
        if k is not None:
            h = k.weight.shape[0]
            start = dst_attn.in_proj_weight.shape[0] // 3
            with torch.no_grad():
                dst_attn.in_proj_weight[start : start + h].copy_(k.weight)
        # v_proj -> last H rows
        v = getattr(src_attn, "v_proj", None)
        if v is not None:
            h = v.weight.shape[0]
            start = 2 * (dst_attn.in_proj_weight.shape[0] // 3)
            with torch.no_grad():
                dst_attn.in_proj_weight[start : start + h].copy_(v.weight)
        # o_proj -> out_proj
        o = getattr(src_attn, "o_proj", None)
        if o is not None:
            with torch.no_grad():
                dst_attn.out_proj.weight.copy_(o.weight)

    src_mlp = getattr(src, "mlp", None)
    if src_mlp is not None:
        gate = getattr(src_mlp, "gate_proj", None)
        down = getattr(src_mlp, "down_proj", None)
        if gate is not None:
            with torch.no_grad():
                # Approximate: use the gate projection as the single FFN expansion.
                dst.linear1.weight[: gate.weight.shape[0]].copy_(gate.weight)
        if down is not None:
            with torch.no_grad():
                dst.linear2.weight.copy_(down.weight)

    # Layer norms
    src_ln1 = getattr(src, "input_layernorm", None)
    if src_ln1 is not None:
        with torch.no_grad():
            dst.norm1.weight.copy_(src_ln1.weight)
    src_ln2 = getattr(src, "post_attention_layernorm", None)
    if src_ln2 is not None:
        with torch.no_grad():
            dst.norm2.weight.copy_(src_ln2.weight)


class HiddenStatePreservationLoss(nn.Module):
    """Loss that encourages compressed hidden states to preserve teacher states.

    Supports three modes:
      * ``mse``: mean-squared error.
      * ``cosine``: ``1 - cosine_similarity(h_student, h_teacher)``.
      * ``contrastive``: a simple InfoNCE-style loss where the teacher vector at
        the same position is the positive example and all other positions in the
        batch are negatives.

    Both ``student`` and ``teacher`` may be a single hidden-state tensor or a
    list of tensors (e.g. one per target layer).  When sequence lengths differ
    the teacher sequence is adaptively pooled to match the student sequence.
    Per-layer losses are averaged unless ``layer_weights`` is supplied.
    """

    def __init__(
        self,
        loss_type: str = "mse",
        layer_weights: Optional[torch.Tensor] = None,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.loss_type = loss_type.lower()
        if self.loss_type not in ("mse", "cosine", "contrastive"):
            raise ValueError(
                f"Unknown loss_type={loss_type!r}; expected mse, cosine, or contrastive"
            )
        self.layer_weights = layer_weights
        self.temperature = temperature

    def forward(
        self,
        student: Union[torch.Tensor, List[torch.Tensor]],
        teacher: Union[torch.Tensor, List[torch.Tensor]],
    ) -> torch.Tensor:
        if isinstance(student, torch.Tensor):
            student = [student]
        if isinstance(teacher, torch.Tensor):
            teacher = [teacher]
        if len(student) != len(teacher):
            raise ValueError(
                f"student and teacher must have the same number of layers, "
                f"got {len(student)} and {len(teacher)}"
            )

        losses = []
        for s, t in zip(student, teacher):
            losses.append(self._layer_loss(s, t))
        losses = torch.stack(losses)

        if self.layer_weights is not None:
            if self.layer_weights.shape != losses.shape:
                raise ValueError(
                    f"layer_weights shape {self.layer_weights.shape} does not match "
                    f"losses shape {losses.shape}"
                )
            return (
                losses * self.layer_weights
            ).sum() / self.layer_weights.sum().clamp_min(1e-6)
        return losses.mean()

    def _align_teacher(
        self, teacher: torch.Tensor, student: torch.Tensor
    ) -> torch.Tensor:
        """Pool teacher sequence to match student sequence length."""
        if teacher.shape[1] == student.shape[1]:
            return teacher
        # Adaptive average pooling groups teacher tokens into student-length bins.
        # This is equivalent to mean-pooling each compress_ratio group when the
        # ratio is an integer.
        pooled = F.adaptive_avg_pool1d(
            teacher.transpose(1, 2), student.shape[1]
        ).transpose(1, 2)
        return pooled

    def _layer_loss(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        teacher = self._align_teacher(teacher, student)
        if self.loss_type == "mse":
            return F.mse_loss(student, teacher)
        if self.loss_type == "cosine":
            sim = F.cosine_similarity(student, teacher, dim=-1)
            return (1.0 - sim).mean()
        # contrastive: InfoNCE over flattened positions.
        s = F.normalize(student.view(-1, student.shape[-1]), dim=-1)
        t = F.normalize(teacher.view(-1, teacher.shape[-1]), dim=-1)
        logits = torch.matmul(s, t.t()) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels)


def get_compressor(
    config,
    compressor_type: Optional[str] = None,
    target_model: Optional[nn.Module] = None,
):
    """Factory for DFlash prefix compressors.

    Compressor hyper-parameters are read from ``config.dflash_config`` when
    ``compressor_type`` is not provided explicitly.
    """
    dflash_config = getattr(config, "dflash_config", {}) or {}
    compressor_type = compressor_type or dflash_config.get("compressor_type", "mlp")
    compress_ratio = dflash_config.get("compress_ratio", 2)
    early_layers = dflash_config.get("compressor_early_layers", 0)

    if compressor_type in ("linear", "mlp"):
        return EmbeddingPairCompressor(
            config=config,
            compress_ratio=compress_ratio,
            compressor_type=compressor_type,
        )
    if compressor_type == "early_semantic":
        return EarlySemanticCompressor(
            config=config,
            early_layers=early_layers,
            compress_ratio=compress_ratio,
            compressor_type="mlp",
            target_model=target_model,
        )
    raise ValueError(f"Unknown compressor_type={compressor_type!r}")
