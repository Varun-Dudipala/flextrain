"""Randomized fault injection: kill random ranks at random moments and check nothing was lost.

Each trial launches the 2-rank toy job under the elastic agent and delivers ``SIGKILL`` to a
randomly chosen rank once training passes a randomly chosen step - up to ``max_kills`` times per
trial, each kill aimed at the incarnation that the previous kill caused. The kills come from
outside the job and are not aligned with anything: they land mid-step, inside a collective,
while the background writer is in the middle of a checkpoint, or while the job is finishing.

A trial passes only if, after all of that, the job

* exits 0 with every rank ``completed`` at the final step,
* was restarted exactly once per delivered kill,
* resumed from a checkpoint no older than the kill allows (a restart from scratch would also
  reach the right weights, since training is deterministic - but it would be losing progress),
* trained each optimizer step on exactly the samples an uninterrupted run uses, and
* ends with weights that are **bitwise identical** to an uninterrupted run on every rank.

Run a campaign (prints one line per trial and a summary; non-zero exit if any trial fails):

    python -m tests.integration.chaos --trials 50 --max-kills 3 --output chaos_results.json

Trials are reproducible: trial ``i`` of a campaign with ``--seed S`` always draws the same kill
plan (which ranks, at which steps). The exact instant a kill lands still varies from run to run,
which is the point.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from tests.integration.harness import (
    DATASET_SIZE,
    GLOBAL_BATCH,
    expected_permutation,
    final_step_samples,
    job_env,
    launch_cmd,
    read_records,
    results,
    run_job,
    write_config,
)

MAX_STEPS = 60  # 5 epochs of the 96-sample toy dataset at a global batch of 8
STEP_SLEEP = "0.02"  # slows the job down enough for kills to land throughout it
SAVE_INTERVAL = 5  # checkpoint cadence of the toy config (see harness.write_config)
NUM_RANKS = 2
# Collective timeout. Where a dead peer makes the survivor's collective fail at once (Linux) it
# is irrelevant; where the survivor blocks instead (macOS) it bounds how long each kill costs.
TIMEOUT_MINUTES = 0.15


@dataclass
class Kill:
    rank: int
    planned_step: int
    incarnation: int
    delivered_at_step: Optional[int] = None  # None: the job finished before this kill was due


@dataclass
class TrialResult:
    seed: int
    passed: bool
    kills: List[Kill] = field(default_factory=list)
    restarts: Optional[int] = None
    seconds: float = 0.0
    failures: List[str] = field(default_factory=list)

    @property
    def delivered(self) -> int:
        return sum(k.delivered_at_step is not None for k in self.kills)


def chaos_config(tmp: Path) -> Path:
    return write_config(tmp, training__max_steps=MAX_STEPS, distributed__timeout_minutes=TIMEOUT_MINUTES)


def reference_digest(tmp: Path) -> str:
    """Weights digest of an uninterrupted run - what every trial must reproduce bit for bit."""
    out = tmp / "out"
    proc = run_job(chaos_config(tmp), tmp, out, nproc=NUM_RANKS)
    if proc.returncode != 0:
        raise RuntimeError(f"reference run failed:\n{proc.stderr[-4000:]}")
    res = results(out)
    if res[0]["digest"] != res[1]["digest"] or res[0]["step"] != MAX_STEPS:
        raise RuntimeError("reference run is not self-consistent")
    return res[0]["digest"]


def plan_kills(rng: random.Random, max_kills: int) -> List[Kill]:
    count = rng.randint(1, max_kills)
    steps = sorted(rng.sample(range(1, MAX_STEPS), count))
    return [Kill(rank=rng.randrange(NUM_RANKS), planned_step=step, incarnation=i) for i, step in enumerate(steps)]


def _progress(out: Path, rank: int) -> tuple:
    """(incarnation, step) of the latest micro-batch ``rank`` logged, or (-1, 0)."""
    records = read_records(out / f"samples_rank{rank}.jsonl")
    return (records[-1]["restart"], records[-1]["step"]) if records else (-1, 0)


DELIVERY_DEADLINE_SECONDS = 180  # an incarnation that takes longer than this to reach its kill is hung


def _deliver(kill: Kill, out: Path, proc: subprocess.Popen) -> None:
    """SIGKILL ``kill.rank`` once the incarnation this kill is aimed at has passed the planned step."""
    deadline = time.monotonic() + DELIVERY_DEADLINE_SECONDS  # monotonic: unaffected by system sleep
    while proc.poll() is None:
        if time.monotonic() > deadline:
            raise TimeoutError(f"incarnation {kill.incarnation} never reached step {kill.planned_step}")
        incarnation, step = _progress(out, kill.rank)
        if incarnation > kill.incarnation:
            return  # that incarnation is already gone (it cannot be: we kill each one once) - give up
        if incarnation == kill.incarnation and step >= kill.planned_step:
            try:
                os.kill(int((out / f"pid_rank{kill.rank}").read_text()), signal.SIGKILL)
            except (ProcessLookupError, ValueError, FileNotFoundError):
                return  # exited on its own between our check and the kill
            kill.delivered_at_step = step
            return
        time.sleep(0.005)


def _check(trial: TrialResult, tmp: Path, out: Path, returncode: int, reference: str) -> None:
    fail = trial.failures.append
    if returncode != 0:
        fail(f"launcher exited with code {returncode}")
    res = results(out)
    if sorted(res) != list(range(NUM_RANKS)):
        fail(f"results written by ranks {sorted(res)}, expected all {NUM_RANKS}")
        return
    trial.restarts = res[0]["restart"]
    for rank, r in res.items():
        if (r["status"], r["step"]) != ("completed", MAX_STEPS):
            fail(f"rank {rank} ended {r['status']} at step {r['step']}")
        if r["digest"] != reference:
            fail(f"rank {rank} weights differ from the uninterrupted run")
    if trial.restarts != trial.delivered:
        fail(f"{trial.delivered} kill(s) delivered but the group restarted {trial.restarts} time(s)")

    # Progress must survive the kill. When the last kill was delivered the killed rank had logged
    # a micro-batch of step s, so steps < s were done; any checkpoint taken at or before step s - 3
    # had at least two further steps to become durable. The final incarnation must resume from
    # that checkpoint or a newer one.
    delivered = [k.delivered_at_step for k in trial.kills if k.delivered_at_step is not None]
    if delivered:
        must_have = SAVE_INTERVAL * ((delivered[-1] - 3) // SAVE_INTERVAL)
        match = re.search(r"checkpoint_step(\d+)\.pt$", res[0]["resumed_from"] or "")
        resumed_step = int(match.group(1)) if match else 0
        if resumed_step < must_have:
            fail(f"killed at step {delivered[-1]} but resumed from step {resumed_step}; "
                 f"the checkpoint of step {must_have} should have been used")

    steps_per_epoch = DATASET_SIZE // GLOBAL_BATCH
    trained_on = final_step_samples(out)
    for step in range(1, MAX_STEPS + 1):
        epoch, k = divmod(step - 1, steps_per_epoch)
        expected = sorted(expected_permutation(epoch)[k * GLOBAL_BATCH : (k + 1) * GLOBAL_BATCH])
        if sorted(trained_on.get(step, [])) != expected:
            fail(f"step {step} trained on the wrong samples")
            break

    records = [json.loads(p.read_text()) for p in (tmp / "home" / "runs").glob("*.json")]
    if len(records) != 1 or records[0].get("status") != "completed":
        fail(f"registry does not show one completed run: {[r.get('status') for r in records]}")


def run_trial(seed: int, reference: str, max_kills: int = 3, workdir: Optional[Path] = None,
              keep_failed: bool = False) -> TrialResult:
    rng = random.Random(seed)
    trial = TrialResult(seed=seed, passed=False, kills=plan_kills(rng, max_kills))
    tmp = Path(tempfile.mkdtemp(prefix=f"chaos-{seed}-", dir=workdir))
    out = tmp / "out"
    started = time.monotonic()
    # stderr goes to a file, not a pipe: nothing reads a pipe while kills are being delivered, and
    # a job blocked on a full pipe would look exactly like the hang this harness exists to find.
    stderr_path = tmp / "launcher_stderr.txt"
    with open(stderr_path, "w") as stderr_file:
        proc = subprocess.Popen(launch_cmd(chaos_config(tmp), nproc=NUM_RANKS, max_restarts=len(trial.kills)),
                                env=job_env(tmp, out, FT_STEP_SLEEP=STEP_SLEEP),
                                stdout=subprocess.DEVNULL, stderr=stderr_file)
        try:
            for kill in trial.kills:
                _deliver(kill, out, proc)
            proc.wait(timeout=120 + 60 * len(trial.kills))
            _check(trial, tmp, out, proc.returncode, reference)
        except (subprocess.TimeoutExpired, TimeoutError) as e:
            trial.failures.append(f"job did not finish in time (hung): {e}")
        finally:
            if proc.poll() is None:
                proc.kill()  # the agent watches its launcher and stops the workers itself
                proc.wait()
    if trial.failures:
        for pid_file in out.glob("pid_rank*"):  # a failed trial must not leave workers behind
            try:
                os.kill(int(pid_file.read_text()), signal.SIGKILL)
            except (ProcessLookupError, ValueError):
                pass
    trial.seconds = round(time.monotonic() - started, 1)
    trial.passed = not trial.failures
    if trial.passed or not keep_failed:
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        trial.failures.append(f"kept for inspection: {tmp}")
    return trial


def describe(trial: TrialResult) -> str:
    kills = ", ".join(
        f"rank {k.rank} @ step {k.delivered_at_step}" if k.delivered_at_step is not None
        else f"rank {k.rank} @ step {k.planned_step} (job finished first)"
        for k in trial.kills)
    verdict = "ok  " if trial.passed else "FAIL"
    line = f"{verdict} seed {trial.seed:>4} | {trial.seconds:>5.1f}s | restarts {trial.restarts} | kills: {kills}"
    return line + "".join(f"\n       - {f}" for f in trial.failures)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--max-kills", type=int, default=3, help="each trial draws 1..N kills")
    parser.add_argument("--seed", type=int, default=0, help="trial i uses seed SEED + i")
    parser.add_argument("--dir", default=None, help="where to put trial directories (default: system temp)")
    parser.add_argument("--keep-failed", action="store_true", help="keep the directory of a failed trial")
    parser.add_argument("--output", default=None, help="write the full results as JSON here")
    args = parser.parse_args(argv)

    workdir = Path(args.dir) if args.dir else None
    ref_dir = Path(tempfile.mkdtemp(prefix="chaos-reference-", dir=workdir))
    try:
        reference = reference_digest(ref_dir)
    finally:
        shutil.rmtree(ref_dir, ignore_errors=True)
    print(f"reference (uninterrupted) weights: sha256 {reference[:16]}...", flush=True)

    trials: List[TrialResult] = []
    for i in range(args.trials):
        trials.append(run_trial(args.seed + i, reference, args.max_kills, workdir, args.keep_failed))
        print(describe(trials[-1]), flush=True)

    passed = sum(t.passed for t in trials)
    delivered = sum(t.delivered for t in trials)
    by_rank: Dict[int, int] = {r: sum(k.rank == r and k.delivered_at_step is not None for t in trials for k in t.kills)
                               for r in range(NUM_RANKS)}
    print(f"\n{passed}/{len(trials)} trials bitwise identical to the uninterrupted run; "
          f"{delivered} kills delivered ({', '.join(f'rank {r}: {n}' for r, n in by_rank.items())})")
    if args.output:
        import torch

        Path(args.output).write_text(json.dumps({
            "environment": {"platform": platform.platform(), "python": platform.python_version(),
                            "torch": torch.__version__},
            "settings": {"trials": args.trials, "max_kills": args.max_kills, "seed": args.seed,
                         "max_steps": MAX_STEPS, "ranks": NUM_RANKS},
            "reference_digest": reference, "passed": passed, "kills_delivered": delivered,
            "trials": [asdict(t) for t in trials],
        }, indent=2))
        print(f"results written to {args.output}")
    return 0 if passed == len(trials) else 1


if __name__ == "__main__":
    sys.exit(main())
