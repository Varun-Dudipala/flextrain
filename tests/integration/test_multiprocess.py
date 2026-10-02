"""Multi-process tests: real ``torchrun`` workers (gloo/CPU) launched via ``flextrain launch``.

These exercise what unit tests cannot: DDP gradient synchronization, the elastic agent
restarting a failed worker group, preemption signals reaching only one rank, and
resuming a job at a different world size.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import pytest
import torch
import yaml

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
                       reason="requires torch.distributed with gloo"),
]

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
        "elastic": {"max_restarts": 0, "monitor_interval": 0.5},
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
            for line in log.read_text().splitlines():
                rec = json.loads(line)
                current.setdefault(rec["step"], []).extend(rec["idx"])
        by_step.update(current)
    return by_step


def expected_permutation(epoch: int = 0) -> List[int]:
    g = torch.Generator()
    g.manual_seed(SEED + epoch)
    return torch.randperm(DATASET_SIZE, generator=g).tolist()


def assert_ok(proc: subprocess.CompletedProcess) -> None:
    assert proc.returncode == 0, (f"launch failed ({proc.returncode})\n"
                                  f"STDOUT:\n{proc.stdout[-4000:]}\nSTDERR:\n{proc.stderr[-6000:]}")


@pytest.fixture(scope="module")
def baseline(tmp_path_factory) -> Dict[str, dict]:
    """An uninterrupted 2-rank run that the other tests compare against."""
    tmp = tmp_path_factory.mktemp("baseline")
    out = tmp / "out"
    assert_ok(run_job(write_config(tmp), tmp, out, nproc=2))
    return results(out)


def _params(result: dict) -> List[torch.Tensor]:
    return [torch.tensor(p) for p in result["params"]]


def test_ddp_ranks_stay_bitwise_identical(baseline):
    assert set(baseline) == {0, 1}
    assert baseline[0]["status"] == "completed" and baseline[0]["step"] == 12
    assert baseline[0]["accumulation_steps"] == 1  # 8 global = 4 micro x 2 ranks
    assert baseline[0]["digest"] == baseline[1]["digest"]


def test_two_ranks_match_single_process_with_accumulation(baseline, tmp_path):
    """DDP's gradient averaging == single process with 2x accumulation on the same samples."""
    out = tmp_path / "out"
    config = write_config(tmp_path)
    proc = subprocess.run([sys.executable, str(TOY_SCRIPT)], env=job_env(tmp_path, out, FLEXTRAIN_CONFIG=str(config)),
                          capture_output=True, text=True, timeout=180)
    assert_ok(proc)
    single = results(out)[0]
    assert single["world_size"] == 1 and single["accumulation_steps"] == 2
    for a, b in zip(_params(single), _params(baseline[0]), strict=True):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)


def test_worker_crash_is_recovered_by_elastic_restart(baseline, tmp_path):
    out = tmp_path / "out"
    proc = run_job(write_config(tmp_path), tmp_path, out, nproc=2, max_restarts=1, FT_CRASH_AT_STEP="7")
    assert_ok(proc)
    res = results(out)
    assert res[0]["restart"] == 1, "the worker group should have been restarted once"
    assert res[0]["resumed_from"].endswith("checkpoint_step00000005.pt")
    assert res[0]["step"] == 12
    # Resuming replays steps 6-7 exactly: the recovered run is bitwise identical to the uninterrupted one.
    assert res[0]["digest"] == res[1]["digest"] == baseline[0]["digest"]

    registry = json.loads((tmp_path / "home" / "runs").glob("*.json").__next__().read_text())
    assert registry["status"] == "completed" and registry["restart_count"] == 1


def test_crash_without_restarts_fails_the_job(tmp_path):
    out = tmp_path / "out"
    proc = run_job(write_config(tmp_path), tmp_path, out, nproc=2, max_restarts=0, FT_CRASH_AT_STEP="7")
    assert proc.returncode != 0
    ckpts = sorted(p.name for p in (tmp_path / "outputs" / "toy" / "checkpoints").glob("*.pt"))
    assert ckpts == ["checkpoint_step00000005.pt"]
    # Rank 0 owns the registry record; it fails when its peer disappears mid-collective.
    registry = json.loads(next((tmp_path / "home" / "runs").glob("*.json")).read_text())
    assert registry["status"] == "failed" and registry.get("error")


def test_sigterm_to_one_rank_stops_all_ranks_at_the_same_step(tmp_path):
    out = tmp_path / "out"
    config = write_config(tmp_path, training__max_steps=100_000, checkpoint__save_interval_steps=0)
    proc = subprocess.Popen(launch_cmd(config, nproc=2), env=job_env(tmp_path, out, FT_STEP_SLEEP="0.05"),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.time() + 120
        while time.time() < deadline:
            steps = step_samples([out])
            if (out / "pid_rank1").exists() and steps and max(steps) >= 5:
                break
            time.sleep(0.2)
        else:
            pytest.fail("job did not reach step 5")
        os.kill(int((out / "pid_rank1").read_text()), signal.SIGTERM)  # only rank 1 is "preempted"
        stdout, stderr = proc.communicate(timeout=120)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert proc.returncode == 0, stderr[-4000:]
    res = results(out)
    assert res[0]["status"] == res[1]["status"] == "preempted"
    assert res[0]["step"] == res[1]["step"]
    stop_step = res[0]["step"]
    assert 5 <= stop_step < 100_000
    ckpt = tmp_path / "outputs" / "toy" / "checkpoints" / f"checkpoint_step{stop_step:08d}.pt"
    assert ckpt.exists(), "a consistent checkpoint must be written at the agreed stop step"
    registry = json.loads(next((tmp_path / "home" / "runs").glob("*.json")).read_text())
    assert registry["status"] == "preempted"


def test_elastic_resize_resumes_from_exact_global_data_position(baseline, tmp_path):
    """Run on 2 ranks, fail, resume on 1 rank: same global batch, no sample repeated or skipped."""
    config = write_config(tmp_path)
    out_a, out_b = tmp_path / "phase_a", tmp_path / "phase_b"
    proc_a = run_job(config, tmp_path, out_a, nproc=2, max_restarts=0, FT_CRASH_AT_STEP="7")
    assert proc_a.returncode != 0
    proc_b = run_job(config, tmp_path, out_b, nproc=1)
    assert_ok(proc_b)

    res = results(out_b)[0]
    assert res["world_size"] == 1 and res["accumulation_steps"] == 2  # global batch preserved at 8
    assert res["resumed_from"].endswith("checkpoint_step00000005.pt") and res["step"] == 12

    by_step = step_samples([out_a, out_b])
    perm = expected_permutation(epoch=0)
    for step in range(1, 13):
        assert sorted(by_step[step]) == sorted(perm[(step - 1) * GLOBAL_BATCH : step * GLOBAL_BATCH]), step
    consumed = [i for step in range(1, 13) for i in by_step[step]]
    assert sorted(consumed) == list(range(DATASET_SIZE))  # every sample exactly once in the epoch

    for a, b in zip(_params(res), _params(baseline[0]), strict=True):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)
