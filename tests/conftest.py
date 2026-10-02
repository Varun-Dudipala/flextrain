"""Shared fixtures."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from flextrain.config import Config


@pytest.fixture(autouse=True)
def _isolate_environment(tmp_path, monkeypatch):
    """Keep the run registry and launcher env vars out of the user's home / shell."""
    monkeypatch.setenv("FLEXTRAIN_HOME", str(tmp_path / "flextrain_home"))
    for var in ("FLEXTRAIN_CONFIG", "FLEXTRAIN_OVERRIDES", "WORLD_SIZE", "RANK", "LOCAL_RANK",
                "TORCHELASTIC_RESTART_COUNT", "TORCHELASTIC_RUN_ID", "TORCHELASTIC_MAX_RESTARTS"):
        monkeypatch.delenv(var, raising=False)
    yield
    for handler in list(logging.getLogger("flextrain").handlers):
        if isinstance(handler, logging.FileHandler):
            logging.getLogger("flextrain").removeHandler(handler)
            handler.close()


class RegressionDataset(Dataset):
    """Deterministic y = sum(x) regression data; item i is identical in every process."""

    def __init__(self, n: int = 64, dim: int = 8):
        self.n, self.dim = n, dim

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        g = torch.Generator().manual_seed(i)
        x = torch.randn(self.dim, generator=g)
        return {"x": x, "y": x.sum(-1, keepdim=True)}


class TinyModel(nn.Module):
    """Small MLP with dropout (so exact-resume tests also cover RNG state)."""

    def __init__(self, dim: int = 8, hidden: int = 16, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x, y=None):
        out = self.fc2(self.drop(torch.relu(self.fc1(x))))
        loss = ((out - y) ** 2).mean() if y is not None else None
        return out, loss


@pytest.fixture
def dataset():
    return RegressionDataset()


@pytest.fixture
def make_config(tmp_path: Path):
    """Factory for small CPU training configs rooted in ``tmp_path``."""

    def _make(max_steps: int = 20, **sections) -> Config:
        cfg = Config()
        cfg.main.output_dir = str(tmp_path / "outputs")
        cfg.main.log_to_file = False
        cfg.training.batch_size = 4
        cfg.training.gradient_accumulation_steps = 2
        cfg.training.max_steps = max_steps
        cfg.training.learning_rate = 1e-2
        cfg.training.warmup_steps = 2
        cfg.training.num_workers = 0
        cfg.training.log_interval = 5
        cfg.checkpoint.save_interval_steps = 5
        cfg.checkpoint.keep_last_n = 10
        for section, values in sections.items():
            for key, value in values.items():
                setattr(getattr(cfg, section), key, value)
        cfg.validate()
        return cfg

    return _make


def params_equal(a: nn.Module, b: nn.Module) -> bool:
    return all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters(), strict=True))
