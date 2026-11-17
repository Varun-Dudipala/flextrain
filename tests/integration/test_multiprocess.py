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

from tests.integration.harness import (
    DATASET_SIZE,
    GLOBAL_BATCH,
    TOY_SCRIPT,
    expected_permutation,
    job_env,
    launch_cmd,
    process_alive,
    read_records,
    results,
    run_job,
    step_samples,
    write_config,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
                       reason="requires torch.distributed with gloo"),
]


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


def test_divergence_is_rolled_back_on_every_rank_without_a_restart(tmp_path):
    """The data of steps 7-9 produces NaN losses. Steps 7 and 8 are skipped, step 9 trips the
    guard, and both ranks go back to the checkpoint of step 5 and continue with the data of
    step 10 - in the same processes, with no help from the elastic agent."""
    out = tmp_path / "out"
    permutation = expected_permutation(0)
    bad = permutation[6 * GLOBAL_BATCH: 9 * GLOBAL_BATCH]
    config = write_config(tmp_path, fault_tolerance__max_consecutive_nonfinite=2)
    proc = run_job(config, tmp_path, out, nproc=2, FT_BAD_SAMPLES=",".join(map(str, bad)))
    assert_ok(proc)

    res = results(out)
    assert res[0]["restart"] == res[1]["restart"] == 0
    assert res[0]["step"] == res[1]["step"] == 12 and res[0]["status"] == "completed"
    assert res[0]["digest"] == res[1]["digest"], "ranks diverged from each other across the rollback"
    for rank in (0, 1):
        (record,) = res[rank]["rollbacks"]
        assert (record["diverged_at_step"], record["rolled_back_to_step"]) == (9, 5)
        assert record["skipped_data_to"] == {"epoch": 0, "consumed": 9 * GLOBAL_BATCH}

    # What was trained on after the rollback: everything each rank logged after its failed step 9.
    after: List[dict] = []
    for rank in (0, 1):
        records = read_records(out / f"samples_rank{rank}.jsonl")
        failed = max(i for i, rec in enumerate(records) if rec["step"] == 9 and set(rec["idx"]) & set(bad))
        after.extend(records[failed + 1:])
    first_step = sorted(i for rec in after if rec["step"] == 6 for i in rec["idx"])
    assert first_step == sorted(permutation[9 * GLOBAL_BATCH: 10 * GLOBAL_BATCH])
    in_epoch_0 = {i for rec in after if rec["step"] <= 8 for i in rec["idx"]}  # steps 6-8 end epoch 0
    assert not in_epoch_0 & set(permutation[5 * GLOBAL_BATCH: 9 * GLOBAL_BATCH]), "skipped data came back"

    registry = json.loads((tmp_path / "home" / "runs").glob("*.json").__next__().read_text())
    assert registry["status"] == "completed" and registry["rollbacks"] == 1 and registry["restart_count"] == 0


def test_crash_without_restarts_fails_the_job(tmp_path):
    out = tmp_path / "out"
    proc = run_job(write_config(tmp_path), tmp_path, out, nproc=2, max_restarts=0, FT_CRASH_AT_STEP="7")
    assert proc.returncode != 0
    ckpts = sorted(p.name for p in (tmp_path / "outputs" / "toy" / "checkpoints").glob("*.pt"))
    assert ckpts == ["checkpoint_step00000005.pt"]
    # Rank 0 owns the registry record. Where the collective errors out when the peer disappears
    # (Linux) it records its own failure; where it stays blocked until the agent SIGKILLs it
    # (macOS) the launcher records the failure instead. Either way the run must not say "running".
    registry = json.loads(next((tmp_path / "home" / "runs").glob("*.json")).read_text())
    assert registry["status"] == "failed" and registry.get("error")


def test_checkpoint_write_failure_on_rank_0_stops_every_rank_together(tmp_path):
    """Only rank 0 touches storage. When its writes fail, the other ranks must be told - otherwise
    they carry on into the next collective without rank 0 and sit there until a timeout."""
    out = tmp_path / "out"
    proc = run_job(write_config(tmp_path), tmp_path, out, nproc=2, FT_FAIL_SAVE_AT_STEP="5")
    assert proc.returncode != 0
    assert "injected failure: no space left on device" in proc.stderr  # rank 0: the real error
    assert "failed on another rank" in proc.stderr  # rank 1: stopped by agreement, not by a timeout
    assert not list((tmp_path / "outputs" / "toy" / "checkpoints").glob("*.pt"))
    registry = json.loads(next((tmp_path / "home" / "runs").glob("*.json")).read_text())
    assert registry["status"] == "failed" and registry.get("last_checkpoint") is None


def test_sigterm_to_one_rank_stops_all_ranks_at_the_same_step(tmp_path):
    out = tmp_path / "out"
    config = write_config(tmp_path, training__max_steps=100_000, checkpoint__save_interval_steps=0)
    proc = subprocess.Popen(launch_cmd(config, nproc=2), env=job_env(tmp_path, out, FT_STEP_SLEEP="0.05"),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
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


def wait_for_step(out: Path, step: int, timeout: float = 120) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        steps = step_samples([out])
        if (out / "pid_rank1").exists() and steps and max(steps) >= step:
            return
        time.sleep(0.2)
    pytest.fail(f"job did not reach step {step}")


@pytest.mark.parametrize("signame", ["SIGTERM", "SIGINT"])
def test_stop_signal_to_the_launcher_stops_the_whole_job_cleanly(tmp_path, signame):
    """A scheduler (SIGTERM) or a terminal (Ctrl-C) signals `flextrain launch`, not the workers.
    The launcher must pass it on: every rank stops after the same step, one checkpoint is
    written, and no process is left behind."""
    out = tmp_path / "out"
    config = write_config(tmp_path, training__max_steps=100_000, checkpoint__save_interval_steps=0)
    proc = subprocess.Popen(launch_cmd(config, nproc=2), env=job_env(tmp_path, out, FT_STEP_SLEEP="0.05"),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        wait_for_step(out, 5)
        worker_pids = [int((out / f"pid_rank{r}").read_text()) for r in (0, 1)]
        os.kill(proc.pid, getattr(signal, signame))  # the launcher only
        proc.communicate(timeout=120)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert proc.returncode == 0, "a job that stopped on request with a checkpoint is not a failure"
    res = results(out)
    assert res[0]["status"] == res[1]["status"] == "preempted"
    assert res[0]["step"] == res[1]["step"] >= 5
    ckpt = tmp_path / "outputs" / "toy" / "checkpoints" / f"checkpoint_step{res[0]['step']:08d}.pt"
    assert ckpt.exists(), "a consistent checkpoint must be written at the agreed stop step"
    registry = json.loads(next((tmp_path / "home" / "runs").glob("*.json")).read_text())
    assert registry["status"] == "preempted"
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and any(process_alive(pid) for pid in worker_pids):
        time.sleep(0.2)
    assert not any(process_alive(pid) for pid in worker_pids), "workers outlived the launcher"


def test_killing_the_launcher_stops_the_job_instead_of_orphaning_it(tmp_path):
    """SIGKILL cannot be forwarded - the launcher is simply gone. The agent runs in its own
    session, so without a watch on its parent it would keep the workers alive (and keep
    restarting them) with nothing left to stop it. It must notice, and stop them cleanly."""
    out = tmp_path / "out"
    config = write_config(tmp_path, training__max_steps=100_000, checkpoint__save_interval_steps=0)
    proc = subprocess.Popen(launch_cmd(config, nproc=2, max_restarts=3),
                            env=job_env(tmp_path, out, FT_STEP_SLEEP="0.05"),
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    worker_pids = []
    try:
        wait_for_step(out, 5)
        worker_pids = [int((out / f"pid_rank{r}").read_text()) for r in (0, 1)]
        proc.kill()
        proc.wait()
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and any(process_alive(pid) for pid in worker_pids):
            time.sleep(0.2)
        assert not any(process_alive(pid) for pid in worker_pids), "workers outlived a killed launcher"
    finally:
        for pid in worker_pids:
            if process_alive(pid):
                os.kill(pid, signal.SIGKILL)
    res = results(out)
    assert res[0]["status"] == res[1]["status"] == "preempted" and res[0]["step"] == res[1]["step"]
    registry = json.loads(next((tmp_path / "home" / "runs").glob("*.json")).read_text())
    assert registry["status"] == "preempted" and registry["last_checkpoint_step"] == res[0]["step"]


def test_hung_worker_times_out_and_the_job_recovers(tmp_path):
    """Freeze one rank (SIGSTOP: alive, but silent - like a deadlock or a dead NIC). Its peer's
    collective must time out instead of waiting forever, and the elastic agent must then restart
    the group, which resumes from the last checkpoint and finishes."""
    out = tmp_path / "out"
    config = write_config(tmp_path, training__max_steps=120, distributed__timeout_minutes=0.25)
    proc = subprocess.Popen(launch_cmd(config, nproc=2, max_restarts=1),
                            env=job_env(tmp_path, out, FT_STEP_SLEEP="0.05"),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    frozen = None
    try:
        wait_for_step(out, 8)  # past the first checkpoint (step 5)
        frozen = int((out / "pid_rank1").read_text())
        os.kill(frozen, signal.SIGSTOP)
        frozen_at = time.monotonic()
        stdout, stderr = proc.communicate(timeout=170)
    finally:
        if proc.poll() is None:
            proc.kill()
        if frozen is not None and process_alive(frozen):
            os.kill(frozen, signal.SIGKILL)
    assert proc.returncode == 0, stderr[-4000:]
    assert time.monotonic() - frozen_at > 15, "the job cannot have recovered before the 15 s collective timeout"
    res = results(out)
    assert res[0]["restart"] == 1, "the worker group should have been restarted once"
    assert res[0]["status"] == res[1]["status"] == "completed" and res[0]["step"] == 120
    assert "checkpoint_step" in res[0]["resumed_from"]
    assert res[0]["digest"] == res[1]["digest"]
    registry = json.loads(next((tmp_path / "home" / "runs").glob("*.json")).read_text())
    assert registry["status"] == "completed" and registry["restart_count"] == 1


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
