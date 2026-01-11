"""Unit tests for GPT-2 model."""

import pytest
import torch

from examples.gpt2_training.model import (
    GPT2Config,
    GPT2Attention,
    GPT2MLP,
    GPT2Block,
    GPT2LMHeadModel,
    create_gpt2_model,
)


class TestGPT2Config:
    """Tests for GPT2Config."""

    def test_default_values(self):
        config = GPT2Config()
        assert config.vocab_size == 50257
        assert config.hidden_size == 768
        assert config.num_hidden_layers == 12
        assert config.num_attention_heads == 12

    def test_custom_values(self):
        config = GPT2Config(hidden_size=1024, num_hidden_layers=24)
        assert config.hidden_size == 1024
        assert config.num_hidden_layers == 24


class TestGPT2Attention:
    """Tests for GPT2Attention."""

    def test_forward_shape(self):
        config = GPT2Config()
        attn = GPT2Attention(config)

        batch_size, seq_len = 2, 64
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

        output = attn(hidden_states)
        assert output.shape == (batch_size, seq_len, config.hidden_size)

    def test_causal_masking(self):
        config = GPT2Config()
        attn = GPT2Attention(config)

        # The bias should be lower triangular
        seq_len = 32
        mask = attn.bias[0, 0, :seq_len, :seq_len]
        assert torch.allclose(mask, torch.tril(torch.ones(seq_len, seq_len)))


class TestGPT2MLP:
    """Tests for GPT2MLP."""

    def test_forward_shape(self):
        config = GPT2Config()
        mlp = GPT2MLP(config)

        batch_size, seq_len = 2, 64
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

        output = mlp(hidden_states)
        assert output.shape == (batch_size, seq_len, config.hidden_size)


class TestGPT2Block:
    """Tests for GPT2Block."""

    def test_forward_shape(self):
        config = GPT2Config()
        block = GPT2Block(config)

        batch_size, seq_len = 2, 64
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

        output = block(hidden_states)
        assert output.shape == (batch_size, seq_len, config.hidden_size)

    def test_residual_connection(self):
        config = GPT2Config()
        block = GPT2Block(config)

        # With residual connections, output should not be zero even if input is small
        hidden_states = torch.zeros(1, 10, config.hidden_size)
        output = block(hidden_states)
        # Output should have some non-zero values due to layer norm bias
        assert not torch.allclose(output, torch.zeros_like(output))


class TestGPT2LMHeadModel:
    """Tests for GPT2LMHeadModel."""

    def test_forward_without_labels(self):
        config = GPT2Config()
        model = GPT2LMHeadModel(config)

        batch_size, seq_len = 2, 64
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        logits, loss = model(input_ids)

        assert logits.shape == (batch_size, seq_len, config.vocab_size)
        assert loss is None

    def test_forward_with_labels(self):
        config = GPT2Config()
        model = GPT2LMHeadModel(config)

        batch_size, seq_len = 2, 64
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        labels = input_ids.clone()

        logits, loss = model(input_ids, labels=labels)

        assert logits.shape == (batch_size, seq_len, config.vocab_size)
        assert loss is not None
        assert loss.dim() == 0  # Scalar loss

    def test_weight_tying(self):
        config = GPT2Config()
        model = GPT2LMHeadModel(config)

        # lm_head weights should be same object as embedding weights
        assert model.lm_head.weight is model.wte.weight

    def test_count_parameters(self):
        config = GPT2Config()
        model = GPT2LMHeadModel(config)

        param_count = model.count_parameters()
        assert param_count > 0
        assert param_count == sum(p.numel() for p in model.parameters() if p.requires_grad)


class TestCreateGPT2Model:
    """Tests for create_gpt2_model factory."""

    def test_create_small(self):
        model = create_gpt2_model("small")
        assert model.config.hidden_size == 768
        assert model.config.num_hidden_layers == 12

    def test_create_medium(self):
        model = create_gpt2_model("medium")
        assert model.config.hidden_size == 1024
        assert model.config.num_hidden_layers == 24

    def test_invalid_size(self):
        # Should default to small for invalid size
        model = create_gpt2_model("invalid")
        assert model.config.hidden_size == 768
