"""Helpers shared by the multi-process tests and the chaos harness: write a config, launch the
toy job (``toy_train.py``) through ``flextrain launch``, and read back what its workers logged."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
TOY_SCRIPT = Path(__file__).with_name("toy_train.py")
DATASET_SIZE = 96  # must match IndexedDataset in toy_train.py
GLOBAL_BATCH = 8
SEED = 42


def write_config(tmp: Path, **overrides) -> Path:
    cfg = {
        "main": {"experiment_name": "toy", "output_dir": str(tmp / "outputs"), "seed": SEED, "log_to_file": False},
        "training": {"batch_size": 4, "global_batch_size": GLOBAL_BATCH, "learning_rate": 0.05, "max_steps": 12,
                     "log_interval": 1, "num_workers": 0, "optimizer": "adamw"},
        "distributed": {"backend": "gloo", "device": "cpu", "timeout_minutes": 2},
        "checkpoint": {"save_interval_steps": 5, "keep_last_n": 10},
        "elastic": {"max_restarts": 0, "monitor_interval": 0.5, "shutdown_timeout": 20},
        # A toy step takes milliseconds, so a rank silent for 3 s after a stop signal is blocked.
        # (After a peer dies torchrun waits a fixed 30 s for the survivors; this is what frees them.)
        "fault_tolerance": {"stop_grace_seconds": 3},
    }
    for dotted, value in overrides.items():
        section, key = dotted.split("__")
        cfg[section][key] = value
    path = tmp / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def launch_cmd(config: Path, nproc: int, max_restarts: int = 0) -> List[str]:
    return [sys.executable, "-m", "flextrain.cli.main", "launch", "-c", str(config),
            "--nproc-per-node", str(nproc), "--max-restarts", str(max_restarts), str(TOY_SCRIPT)]


def job_env(tmp: Path, out: Path, **extra: str) -> Dict[str, str]:
    env = dict(os.environ)
    env.update(FLEXTRAIN_HOME=str(tmp / "home"), FT_OUT=str(out), PYTHONPATH=str(REPO_ROOT),
               MASTER_ADDR="127.0.0.1", OMP_NUM_THREADS="1", **extra)
    return env


def run_job(config: Path, tmp: Path, out: Path, nproc: int, max_restarts: int = 0, **extra: str):
    proc = subprocess.run(launch_cmd(config, nproc, max_restarts), env=job_env(tmp, out, **extra),
                          capture_output=True, text=True, timeout=180)
    return proc


def results(out: Path) -> Dict[int, dict]:
    return {int(p.stem.split("rank")[1]): json.loads(p.read_text()) for p in out.glob("result_rank*.json")}


def step_samples(out_dirs: List[Path]) -> Dict[int, List[int]]:
    """step -> sample indices, keeping each step's entries from the *latest* run that logged it."""
    by_step: Dict[int, List[int]] = {}
    for out in out_dirs:  # later directories (later incarnations) override earlier ones
        current: Dict[int, List[int]] = {}
        for log in out.glob("samples_rank*.jsonl"):
            for rec in read_records(log):
                current.setdefault(rec["step"], []).extend(rec["idx"])
        by_step.update(current)
    return by_step


def expected_permutation(epoch: int = 0) -> List[int]:
    g = torch.Generator()
    g.manual_seed(SEED + epoch)
    return torch.randperm(DATASET_SIZE, generator=g).tolist()


def read_records(log: Path) -> List[dict]:
    """Parse a worker's JSONL log, skipping a last line that is still being written."""
    records = []
    try:
        lines = log.read_text().splitlines()
    except FileNotFoundError:
        return records
    for line in lines:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def final_step_samples(out: Path) -> Dict[int, List[int]]:
    """step -> sample indices as consumed by the *last* incarnation that ran that step.

    A killed incarnation may have logged steps that were then replayed after the restart (or a
    step only one rank got to); what the model was finally trained on is the latest replay.
    """
    latest: Dict[int, int] = {}
    by_step: Dict[int, List[int]] = {}
    for log in sorted(out.glob("samples_rank*.jsonl")):
        for rec in read_records(log):
            step, restart = rec["step"], rec["restart"]
            if restart > latest.get(step, -1):
                latest[step], by_step[step] = restart, []
            if restart == latest[step]:
                by_step[step].extend(rec["idx"])
    return by_step


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True
