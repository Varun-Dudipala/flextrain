"""Tiny training job driven by the multi-process integration tests (run under torchrun).

Behaviour is controlled by environment variables so the tests can inject failures:

  FLEXTRAIN_CONFIG     config file (set by ``flextrain launch``)
  FT_OUT               output directory for per-rank logs / results
  FT_CRASH_AT_STEP     raise on rank FT_CRASH_RANK after this optimizer step ...
  FT_CRASH_RANK        ... (default 1) - only in the first incarnation (restart count 0)
  FT_STEP_SLEEP        seconds to sleep per step (gives the tests time to send signals)
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from flextrain import Trainer, load_config
from flextrain.elastic import ElasticEnv


class IndexedDataset(Dataset):
    def __init__(self, n: int = 96, dim: int = 8):
        self.n, self.dim = n, dim

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        g = torch.Generator().manual_seed(i)
        x = torch.randn(self.dim, generator=g)
        return {"x": x, "y": x.sum(-1, keepdim=True), "idx": torch.tensor(i)}


class ToyModel(nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 16), nn.Tanh(), nn.Linear(16, 1))

    def forward(self, x, y, idx=None):
        out = self.net(x)
        return out, ((out - y) ** 2).mean()


def params_digest(model: nn.Module) -> str:
    h = hashlib.sha256()
    for p in model.parameters():
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class InstrumentedTrainer(Trainer):
    def __init__(self, *args, out_dir: Path, **kwargs):
        super().__init__(*args, **kwargs)
        self.out_dir = out_dir
        self.restart = ElasticEnv.from_env().restart_count
        self.crash_at = int(os.environ.get("FT_CRASH_AT_STEP", "0"))
        self.crash_rank = int(os.environ.get("FT_CRASH_RANK", "1"))
        self.step_sleep = float(os.environ.get("FT_STEP_SLEEP", "0"))
        self.sample_log = open(out_dir / f"samples_rank{self.rank}.jsonl", "a")
        (out_dir / f"pid_rank{self.rank}").write_text(str(os.getpid()))

    def compute_loss(self, batch):
        record = {"step": self.global_step + 1, "restart": self.restart, "world_size": self.world_size,
                  "idx": batch["idx"].tolist()}
        self.sample_log.write(json.dumps(record) + "\n")
        self.sample_log.flush()
        return super().compute_loss(batch)

    def _after_step(self):
        stop = super()._after_step()
        if self.step_sleep:
            time.sleep(self.step_sleep)
        if self.crash_at and self.global_step == self.crash_at and self.restart == 0 and self.rank == self.crash_rank:
            raise RuntimeError(f"injected failure on rank {self.rank} at step {self.global_step}")
        return stop


def main() -> int:
    out_dir = Path(os.environ["FT_OUT"])
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_config()
    torch.manual_seed(0)
    trainer = InstrumentedTrainer(ToyModel(), config, IndexedDataset(), out_dir=out_dir)
    result = trainer.train()
    (out_dir / f"result_rank{trainer.rank}.json").write_text(json.dumps({
        "status": result.status, "step": result.global_step, "world_size": trainer.world_size,
        "accumulation_steps": trainer.accumulation_steps, "restart": trainer.restart,
        "resumed_from": trainer.resumed_from, "digest": params_digest(trainer.model),
        "params": [p.detach().cpu().flatten().tolist() for p in trainer.model.parameters()],
    }))
    trainer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
