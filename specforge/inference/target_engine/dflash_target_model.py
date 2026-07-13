from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from . import compressor as compressor_module
from .base import TargetEngine

# NOTE (Phase B2): this module no longer imports sglang internals. The
# SGLang-version-pinned capture path (ServerArgs / ModelConfig / SGLangRunner +
# the extend/capture forward) lives entirely in
# ``sglang_backend.SGLangCaptureBackend``, shared with the eagle3 engine (one
# copy of the forward + mlp-sync). The SGLang engine below composes it, imported
# lazily inside ``from_pretrained`` so ``import specforge`` stays sglang-agnostic.


@dataclass
class DFlashTargetOutput:
    hidden_states: torch.Tensor  # [batch, seq_len, hidden_size]
    input_ids: torch.Tensor  # [batch, seq_len]
    attention_mask: torch.Tensor  # [batch, seq_len]
    loss_mask: torch.Tensor  # [batch, seq_len]


class DFlashTargetEngine(TargetEngine):
    """DFlash target engine — the algorithm ABC over a frozen target backend.

    DFlash captures the concatenated hidden states of an arbitrary list of
    target layers (``set_capture_layers``) and trains on hard real-token labels,
    so — unlike EAGLE3 — there is no target distribution / vocab map. The generic
    :meth:`TargetEngine.capture` hook dispatches to ``generate_dflash_data``, so
    the extraction is byte-identical to the pre-Phase-B path.
    """

    def __init__(self):
        self.capture_layer_ids = None

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        **kwargs,
    ) -> "DFlashTargetEngine":
        """Initialize the target model backend."""

    @abstractmethod
    def generate_dflash_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        use_compression: bool = False,
        requires_grad: bool = False,
    ) -> DFlashTargetOutput:
        """Generate context hidden states for DFlash training.

        Args:
            input_ids: Token IDs of shape ``[B, T]``.
            attention_mask: Attention mask of shape ``[B, T]``.
            loss_mask: Loss mask of shape ``[B, T]``.
            use_compression: If ``True`` and a compressor is attached, run the
                target model on compressed embeddings (sequence length ``T//r``).
            requires_grad: If ``False`` (default), the forward pass is performed
                under ``torch.no_grad()`` for efficiency. Set ``True`` when
                training the compressor and gradients are required.
        """

    def capture(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        **kwargs,
    ) -> DFlashTargetOutput:
        """Generic extraction entry point (see :meth:`TargetEngine.capture`).

        Dispatches to the DFlash-specific ``generate_dflash_data``. DFlash takes
        no extra extraction kwargs, so any are ignored.
        """
        return self.generate_dflash_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
        )

    def set_capture_layers(self, layer_ids: Optional[List[int]] = None) -> None:
        """Set which layers' hidden states to capture (TargetEngine hook)."""
        self.capture_layer_ids = layer_ids


class SGLangDFlashTargetEngine(DFlashTargetEngine):

    backend = "sglang"

    def __init__(self, backend):  # backend: sglang_backend.SGLangCaptureBackend
        super().__init__()  # capture_layer_ids = None
        self._backend = backend

    @property
    def model_runner(self):
        """Kept for back-compat: the underlying sglang ModelRunner."""
        return self._backend.model_runner

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = False,
        **kwargs,
    ) -> "SGLangDFlashTargetEngine":
        # Lazy import so `import specforge` still works without the pinned sglang:
        # the sglang-version coupling lives entirely in SGLangCaptureBackend, which
        # also unifies the extend/mlp-sync forward this engine used to duplicate.
        from .sglang_backend import SGLangCaptureBackend

        backend = SGLangCaptureBackend.build(
            pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            wrap_eagle3_logits=False,
            **kwargs,
        )
        return cls(backend)

    def set_capture_layers(self, layer_ids: List[int]) -> None:
        super().set_capture_layers(layer_ids)  # records self.capture_layer_ids
        # Some target models expose set_eagle3_layers_to_capture; guard on it.
        self._backend.set_eagle3_capture_layers(layer_ids, if_supported=True)

    @torch.no_grad()
    def generate_dflash_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        use_compression: bool = False,
        requires_grad: bool = False,
    ) -> DFlashTargetOutput:
        # SGLang backend does not support compressor injection yet.
        if use_compression:
            raise NotImplementedError(
                "use_compression=True is not supported for SGLangDFlashTargetEngine"
            )
        data_cache, hidden_states_list = self._backend.extend_dflash(
            input_ids, attention_mask, loss_mask
        )

        # Stack back to batch
        hidden_states = torch.cat([h.unsqueeze(0) for h in hidden_states_list], dim=0)
        input_ids = torch.cat([d[0] for d in data_cache], dim=0)
        attention_mask = torch.cat([d[1] for d in data_cache], dim=0)
        loss_mask = torch.cat([d[2] for d in data_cache], dim=0)

        return DFlashTargetOutput(
            hidden_states=hidden_states,
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
        )


class HFDFlashTargetEngine(DFlashTargetEngine):

    backend = "hf"

    def __init__(self, model: nn.Module, compressor: Optional[nn.Module] = None):
        super().__init__()
        self.model = model
        self.compressor = compressor

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = True,
        **kwargs,
    ) -> "HFDFlashTargetEngine":

        target_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
            output_hidden_states=True,
            trust_remote_code=trust_remote_code,
            **kwargs,
        ).eval()

        if device:
            target_model = target_model.to(device)

        return cls(target_model)

    def generate_dflash_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        use_compression: bool = False,
        requires_grad: bool = False,
    ) -> DFlashTargetOutput:
        with torch.set_grad_enabled(requires_grad):
            if use_compression and self.compressor is not None:
                # Compress the prefix embeddings and run the target model on the
                # shorter sequence.  The returned hidden states therefore have
                # length ``T // compress_ratio`` and must be handled accordingly
                # by the caller.
                embeds = self._get_input_embeddings(input_ids)
                compressed_embeds = self.compressor(embeds)
                compressed_attention_mask = self._compress_mask(attention_mask)
                compressed_loss_mask = self._compress_mask(loss_mask)

                outputs = self.model(
                    inputs_embeds=compressed_embeds,
                    attention_mask=compressed_attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
                hidden_states = self._select_hidden_states(outputs.hidden_states)

                return DFlashTargetOutput(
                    hidden_states=hidden_states,
                    input_ids=input_ids,
                    attention_mask=compressed_attention_mask,
                    loss_mask=compressed_loss_mask,
                )

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            hidden_states = self._select_hidden_states(outputs.hidden_states)

            return DFlashTargetOutput(
                hidden_states=hidden_states,
                input_ids=input_ids,
                attention_mask=attention_mask,
                loss_mask=loss_mask,
            )

    def _get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Fetch token embeddings in an architecture-agnostic way."""
        if hasattr(self.model, "get_input_embeddings"):
            return self.model.get_input_embeddings()(input_ids)
        # Common fallback for CausalLM wrappers.
        return self.model.model.embed_tokens(input_ids)

    def _compress_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Compress a mask to match the compressor output length."""
        return compressor_module.compress_mask(mask, self.compressor.compress_ratio)

    def _select_hidden_states(self, hidden_states: tuple[torch.Tensor]) -> torch.Tensor:
        """Select captured layer hidden states or fall back to the last layer."""
        # hidden_states[0] = embedding output; hidden_states[i+1] = layer i output
        offset = 1
        if self.capture_layer_ids is not None:
            selected = []
            for idx in self.capture_layer_ids:
                selected.append(hidden_states[idx + offset])
            return torch.cat(selected, dim=-1)
        return hidden_states[-1]


def get_dflash_target_model(
    pretrained_model_name_or_path: str,
    backend: str = "sglang",
    torch_dtype: torch.dtype = None,
    device: str = None,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> DFlashTargetEngine:
    if backend == "sglang":
        return SGLangDFlashTargetEngine.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            device=device,
            cache_dir=cache_dir,
            **kwargs,
        )
    elif backend == "hf":
        return HFDFlashTargetEngine.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            device=device,
            cache_dir=cache_dir,
            **kwargs,
        )
    else:
        raise ValueError(f"Invalid backend: {backend}")


# --- Back-compat aliases (pre-Phase-B names) -------------------------------
# See the note in eagle3_target_model.py: the ``*TargetModel`` -> ``*TargetEngine``
# rename is import-compatible; these aliases keep existing callers working.
DFlashTargetModel = DFlashTargetEngine
SGLangDFlashTargetModel = SGLangDFlashTargetEngine
HFDFlashTargetModel = HFDFlashTargetEngine
