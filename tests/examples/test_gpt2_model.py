"""Tests for the example GPT-2 model and training script."""

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from examples.gpt2_training.model import (
    GPT2MLP,
    GPT2Attention,
    GPT2Block,
    GPT2Config,
    GPT2LMHeadModel,
    create_gpt2_model,
)

TINY = dict(hidden_size=64, num_hidden_layers=2, num_attention_heads=4, intermediate_size=128,
            vocab_size=100, max_position_embeddings=64)


@pytest.fixture
def config():
    return GPT2Config(**TINY)


def test_default_config_is_gpt2_small():
    cfg = GPT2Config()
    assert (cfg.vocab_size, cfg.hidden_size, cfg.num_hidden_layers, cfg.num_attention_heads) == (50257, 768, 12, 12)


def test_heads_must_divide_hidden_size():
    with pytest.raises(ValueError):
        GPT2Config(hidden_size=100, num_attention_heads=12)


@pytest.mark.parametrize("module_cls", [GPT2Attention, GPT2MLP, GPT2Block])
def test_submodule_shapes(config, module_cls):
    x = torch.randn(2, 16, config.hidden_size)
    assert module_cls(config)(x).shape == x.shape


def test_attention_is_causal(config):
    """Changing a future token must not change the outputs at earlier positions."""
    model = GPT2LMHeadModel(config).eval()
    ids = torch.randint(0, config.vocab_size, (1, 16))
    changed = ids.clone()
    changed[0, 10] = (changed[0, 10] + 1) % config.vocab_size
    with torch.no_grad():
        a, _ = model(ids)
        b, _ = model(changed)
    torch.testing.assert_close(a[:, :10], b[:, :10])
    assert not torch.allclose(a[:, 10:], b[:, 10:])


def test_no_attention_mask_buffers_in_state_dict(config):
    assert all("bias" not in k or k.endswith(".bias") for k in GPT2LMHeadModel(config).state_dict())
    assert not any(k.endswith("attn.bias") and v.dim() == 4 for k, v in GPT2LMHeadModel(config).state_dict().items())


def test_forward_with_and_without_labels(config):
    model = GPT2LMHeadModel(config)
    ids = torch.randint(0, config.vocab_size, (2, 16))
    logits, loss = model(ids)
    assert logits.shape == (2, 16, config.vocab_size) and loss is None
    _, loss = model(ids, labels=ids)
    assert loss.dim() == 0 and loss.item() == pytest.approx(torch.log(torch.tensor(100.0)).item(), rel=0.2)


def test_sequence_longer_than_context_rejected(config):
    with pytest.raises(ValueError):
        GPT2LMHeadModel(config)(torch.zeros(1, 65, dtype=torch.long))


def test_weight_tying(config):
    model = GPT2LMHeadModel(config)
    assert model.lm_head.weight is model.wte.weight


def test_count_parameters(config):
    model = GPT2LMHeadModel(config)
    assert model.count_parameters() == sum(p.numel() for p in model.parameters())


def test_presets():
    assert create_gpt2_model("tiny").config.hidden_size == 128
    small = create_gpt2_model("small")
    assert 124e6 < small.count_parameters() < 125e6  # GPT-2 small: 124M with tied embeddings
    with pytest.raises(ValueError, match="unknown GPT-2 size"):
        create_gpt2_model("invalid")


def test_example_script_runs_end_to_end(tmp_path):
    script = Path(__file__).resolve().parents[2] / "examples" / "gpt2_training" / "train_gpt2.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--set", "training.max_steps=4", "--set", "training.warmup_steps=0",
         "--set", f"main.output_dir={tmp_path}",
         "--set", "training.eval_interval=2", "--set", "training.eval_max_batches=1"],
        capture_output=True, text=True, timeout=300, env={**__import__("os").environ, "FLEXTRAIN_HOME": str(tmp_path)},
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "completed at step 4" in proc.stdout
    assert (tmp_path / "gpt2-bytes" / "checkpoints" / "checkpoint_step00000004.pt").exists()
