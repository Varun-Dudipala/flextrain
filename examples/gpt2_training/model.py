"""Minimal GPT-2 (decoder-only transformer) used by the example and the benchmark."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPT2Config:
    vocab_size: int = 50257
    max_position_embeddings: int = 1024
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")


PRESETS = {
    # tiny: CPU-friendly preset for the demo and tests
    "tiny": dict(hidden_size=128, num_hidden_layers=2, num_attention_heads=4, intermediate_size=512,
                 max_position_embeddings=256),
    "small": dict(),  # 124M
    "medium": dict(hidden_size=1024, num_hidden_layers=24, num_attention_heads=16, intermediate_size=4096),
    "large": dict(hidden_size=1280, num_hidden_layers=36, num_attention_heads=20, intermediate_size=5120),
}


class GPT2Attention(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.dropout = config.attention_dropout_prob
        self.c_attn = nn.Linear(self.hidden_size, 3 * self.hidden_size)
        self.c_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.resid_dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, _ = hidden_states.size()
        q, k, v = self.c_attn(hidden_states).split(self.hidden_size, dim=2)
        q, k, v = (t.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2) for t in (q, k, v))
        # Fused causal attention (FlashAttention / memory-efficient kernels where available);
        # no O(max_len^2) mask buffer is stored in the module or its checkpoints.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                             dropout_p=self.dropout if self.training else 0.0)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_size)
        return self.resid_dropout(self.c_proj(out))


class GPT2MLP(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(config.hidden_size, config.intermediate_size)
        self.c_proj = nn.Linear(config.intermediate_size, config.hidden_size)
        self.act = nn.GELU(approximate="tanh")
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.act(self.c_fc(hidden_states))))


class GPT2Block(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.ln_1(hidden_states))
        return hidden_states + self.mlp(self.ln_2(hidden_states))


class GPT2LMHeadModel(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.wpe = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.drop = nn.Dropout(config.hidden_dropout_prob)
        self.h = nn.ModuleList([GPT2Block(config) for _ in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # weight tying

        self.apply(self._init_weights)
        # GPT-2 scales residual projections by 1/sqrt(2 * n_layers) to keep activations stable.
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=config.initializer_range / math.sqrt(2 * config.num_hidden_layers))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        _, seq_length = input_ids.size()
        if seq_length > self.config.max_position_embeddings:
            raise ValueError(f"sequence length {seq_length} > max_position_embeddings "
                             f"{self.config.max_position_embeddings}")
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        hidden_states = self.drop(self.wte(input_ids) + self.wpe(position_ids))
        for block in self.h:
            hidden_states = block(hidden_states)
        logits = self.lm_head(self.ln_f(hidden_states))

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)).float(), shift_labels.view(-1))
        return logits, loss

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_gpt2_model(size: str = "small", **overrides) -> GPT2LMHeadModel:
    if size not in PRESETS:
        raise ValueError(f"unknown GPT-2 size {size!r}; choose from {sorted(PRESETS)}")
    return GPT2LMHeadModel(GPT2Config(**{**PRESETS[size], **overrides}))
