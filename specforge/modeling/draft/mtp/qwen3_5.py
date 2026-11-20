# coding=utf-8
"""Native Qwen3.5 MTP modules used for frozen feature capture."""

import copy
from typing import Optional, Tuple

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    FlashAttentionKwargs,
    GradientCheckpointingLayer,
    Qwen3Config,
    Qwen3MLP,
    Qwen3PreTrainedModel,
    eager_attention_forward,
    rotate_half,
)
from typing_extensions import Unpack


class Qwen3_5RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = hidden_states * (1.0 + self.weight.float())
        return hidden_states.to(input_dtype)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
    return torch.cat([q_embed, q_pass], dim=-1), torch.cat([k_embed, k_pass], dim=-1)


class PartialRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config, head_dim):
        super().__init__()
        rope_theta = getattr(config, "rope_theta", 10000.0)
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        dim = int(head_dim * partial_rotary_factor)
        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)


class Qwen3MTPAttention(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.attn_output_gate = getattr(config, "attn_output_gate", False)
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads
            * self.head_dim
            * (2 if self.attn_output_gate else 1),
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len, _ = hidden_states.size()
        if self.attn_output_gate:
            query_states, gate = torch.chunk(
                self.q_proj(hidden_states).view(bsz, q_len, -1, self.head_dim * 2),
                2,
                dim=-1,
            )
            gate = gate.reshape(bsz, q_len, -1)
        else:
            query_states = self.q_proj(hidden_states).view(
                bsz, q_len, -1, self.head_dim
            )
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = query_states.view(
            bsz, q_len, self.config.num_attention_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.config.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.config.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
        attn_fn = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attn_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, attn_weights = attn_fn(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1)
        if self.attn_output_gate:
            attn_output = attn_output * torch.sigmoid(gate)
        return self.o_proj(attn_output), attn_weights


class Qwen3MTPDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3MTPAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        return outputs


def _causal_mask(batch_size, seq_len, dtype, device):
    mask = torch.full(
        (seq_len, seq_len), torch.finfo(dtype).min, dtype=dtype, device=device
    ).triu(1)
    return mask[None, None].expand(batch_size, 1, -1, -1)


class Qwen3_5MTPModel(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        mtp_config = copy.deepcopy(config)
        mtp_config.num_hidden_layers = 1
        self.layers = nn.ModuleList([Qwen3MTPDecoderLayer(mtp_config, layer_idx=0)])
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = PartialRotaryEmbedding(
            mtp_config,
            getattr(
                mtp_config,
                "head_dim",
                mtp_config.hidden_size // mtp_config.num_attention_heads,
            ),
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        normed_emb = self.pre_fc_norm_embedding(inputs_embeds)
        normed_hidden = self.pre_fc_norm_hidden(hidden_states)
        hidden_states = self.fc(torch.cat([normed_emb, normed_hidden], dim=-1))
        bsz, seq_len, _ = hidden_states.size()
        if position_ids is None:
            position_ids = (
                torch.arange(seq_len, dtype=torch.long, device=hidden_states.device)
                .unsqueeze(0)
                .expand(bsz, -1)
            )
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        if attention_mask is not None and attention_mask.dim() == 2:
            combined_mask = _causal_mask(
                bsz, seq_len, hidden_states.dtype, hidden_states.device
            )
            expanded_mask = (
                1.0 - attention_mask[:, None, None, :].to(hidden_states.dtype)
            ) * torch.finfo(hidden_states.dtype).min
            attention_mask = expanded_mask + combined_mask
        elif (
            attention_mask is None
            and seq_len > 1
            and self.config._attn_implementation != "sdpa"
        ):
            attention_mask = _causal_mask(
                bsz, seq_len, hidden_states.dtype, hidden_states.device
            )
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                **(
                    {"is_causal": True}
                    if attention_mask is None
                    and self.config._attn_implementation == "sdpa"
                    else {}
                ),
            )[0]
        return self.norm(hidden_states)


class Qwen3_5MTPDraftModel(Qwen3PreTrainedModel):
    config_class = Qwen3Config
    _no_split_modules = ["Qwen3MTPDecoderLayer"]

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__(config)
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.mtp = Qwen3_5MTPModel(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_output_embeddings(self):
        return self.mtp.lm_head

    def set_output_embeddings(self, value):
        self.mtp.lm_head = value

    def forward_hidden(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        return self.mtp(
            inputs_embeds=self.embed_tokens(input_ids),
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        draft_hidden = self.forward_hidden(
            input_ids=input_ids,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return CausalLMOutputWithPast(logits=self.mtp.lm_head(draft_hidden))
