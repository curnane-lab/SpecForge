"""Frozen native-Qwen MTP features aligned with canonical DMTP records."""

import torch
from torch import nn
from transformers import AutoConfig
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5TextModel,
    Qwen3_5TextRotaryEmbedding,
)

from specforge.modeling.target.checkpoint import load_checkpoint_tensors

MTP_ALIGNMENT = "anchor_post_norm_v1"
MTP_VERSION = 1


class _NativeMTPModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        self.text = Qwen3_5TextModel(config)

    def forward_hidden(self, input_ids, target_hidden, *, position_ids=None):
        embedded = self.embedding_norm(self.text.embed_tokens(input_ids))
        target = self.hidden_norm(target_hidden)
        fused = self.fc(torch.cat((embedded, target), dim=-1))
        return self.text(
            inputs_embeds=fused,
            position_ids=position_ids,
            use_cache=False,
        ).last_hidden_state


def _native_config(path, *, cache_dir=None):
    native = AutoConfig.from_pretrained(
        path, trust_remote_code=False, cache_dir=cache_dir
    )
    text = getattr(native, "text_config", native)
    if text.model_type != "qwen3_5_text" or getattr(text, "num_experts", 0):
        raise ValueError(
            "mtp_hidden_states capture supports dense Qwen3.5/3.6 text MTP"
        )
    if getattr(native, "quantization_config", None) is not None:
        raise ValueError("mtp_hidden_states capture requires unquantized MTP weights")
    if getattr(text, "mtp_num_hidden_layers", 1) != 1:
        raise ValueError("mtp_hidden_states capture requires one native MTP layer")
    if getattr(text, "mtp_use_dedicated_embeddings", False):
        raise ValueError("dedicated MTP embeddings are not supported")
    rope = getattr(text, "rope_parameters", None) or {}
    if rope.get("rope_type", "default") != "default":
        raise ValueError("mtp_hidden_states capture requires default native RoPE")
    values = text.to_dict()
    values.update(
        num_hidden_layers=1,
        layer_types=["full_attention"],
        use_cache=False,
    )
    config = Qwen3_5TextConfig(**values)
    config._attn_implementation = "sdpa"
    return config


class FrozenMTPCapture:
    def __init__(self, model):
        self.model = model.eval().requires_grad_(False)

    @classmethod
    def from_pretrained(
        cls,
        path,
        *,
        device,
        dtype,
        cache_dir=None,
        embedding_key="model.language_model.embed_tokens.weight",
    ):
        config = _native_config(path, cache_dir=cache_dir)
        with torch.device("meta"):
            model = _NativeMTPModel(config)
        sources = {}
        for key in model.state_dict():
            if key == "embedding_norm.weight":
                source = "mtp.pre_fc_norm_embedding.weight"
            elif key == "hidden_norm.weight":
                source = "mtp.pre_fc_norm_hidden.weight"
            elif key == "fc.weight":
                source = "mtp.fc.weight"
            elif key == "text.embed_tokens.weight":
                source = embedding_key
            elif key == "text.norm.weight":
                source = "mtp.norm.weight"
            elif key.startswith("text.layers.0."):
                source = "mtp.layers.0." + key.removeprefix("text.layers.0.")
            else:
                raise ValueError(f"unmapped native MTP tensor: {key}")
            sources[key] = source
        native = load_checkpoint_tensors(
            path, keys=set(sources.values()), cache_dir=cache_dir
        )
        model.load_state_dict(
            {key: native[source] for key, source in sources.items()},
            strict=True,
            assign=True,
        )
        model.text.rotary_emb = Qwen3_5TextRotaryEmbedding(config)
        model.to(device=device, dtype=dtype)
        model.text.rotary_emb = Qwen3_5TextRotaryEmbedding(config, device=device)
        return cls(model)

    @torch.no_grad()
    def __call__(self, input_ids, target_hidden):
        if input_ids.ndim != 1 or target_hidden.ndim != 2:
            raise ValueError("MTP capture expects one unpadded text sequence")
        if len(input_ids) != len(target_hidden):
            raise ValueError("MTP input_ids and target hidden lengths differ")
        parameters = self.model.text.embed_tokens.weight
        input_ids = input_ids.to(device=parameters.device)
        target_hidden = target_hidden.detach().to(
            device=parameters.device, dtype=parameters.dtype
        )
        output = target_hidden.new_zeros(target_hidden.shape)
        if len(input_ids) > 1:
            output[1:] = self.model.forward_hidden(
                input_ids[None, 1:],
                target_hidden[None, :-1],
                position_ids=torch.arange(len(input_ids) - 1, device=parameters.device)[
                    None
                ],
            )[0]
        return output
