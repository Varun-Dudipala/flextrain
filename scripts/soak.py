#!/usr/bin/env python3
"""Long-run kill test: hours of GPT-2 training under random kills must change nothing.

Two copies of the same 2-rank job (the GPT-2 example, launched with ``flextrain launch``) run
side by side:

* the **reference** is left alone and writes one checkpoint, at the end;
* the **killed** copy checkpoints periodically and has a randomly chosen rank ``SIGKILL``-ed at
  random moments (exponentially distributed gaps), for as long as it runs. The elastic agent
  restarts it each time and it resumes from its newest checkpoint.

The test passes only if both finish and their final checkpoints are identical bit for bit:
every model tensor, every optimizer tensor, the schedule, the data position and every rank's
random-number state. The kills come from outside and are aligned with nothing, so over a long
run they land mid-step, inside collectives, during checkpoint writes and during restarts.

Both jobs run from a frozen copy of the code and of the training text, taken when the test
starts, so the repository can be edited while the test runs. (A restart re-imports the code
and re-reads the data; without the copy, an edit during the run would change the job.)

    python scripts/soak.py --steps 900 --ckpt-every 25 --output results/soak.json

A run is reproducible in its kill plan (``--seed``) but not in where the kills land.
"""

from __future__ import annotations

import argparse
import hashlib
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
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "soak"
TRAIN_SCRIPT = Path("examples") / "gpt2_training" / "train_gpt2.py"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# ---------------------------------------------------------------------------- setup
def freeze(work: Path) -> tuple:
    """Copy the code the jobs run, and build their training text from it.

    Returns the copy's location and a SHA-256 over its files (relative paths and contents, in
    sorted order), which identifies the code that was tested whatever the state of the checkout.
    """
    src = work / "src"
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "outputs")
    shutil.copytree(REPO_ROOT / "flextrain", src / "flextrain", ignore=ignore)
    shutil.copytree(REPO_ROOT / "examples", src / "examples", ignore=ignore)
    files = sorted((src / "flextrain").rglob("*.py"))
    (work / "corpus.txt").write_bytes(b"\n".join(f.read_bytes() for f in files))
    sha = hashlib.sha256()
    for path in sorted(p for p in src.rglob("*") if p.is_file()):
        sha.update(str(path.relative_to(src)).encode())
        sha.update(path.read_bytes())
    return src, sha.hexdigest()


def write_config(path: Path, out_dir: Path, corpus: Path, args: argparse.Namespace, save_every: int) -> None:
    config = {
        "main": {"experiment_name": EXPERIMENT, "output_dir": str(out_dir), "seed": args.seed},
        "model": {"size": args.model_size, "vocab_size": 256},
        "data": {"seq_len": args.seq_len, "paths": [str(corpus)]},
        "training": {
            "batch_size": args.batch_size, "global_batch_size": 2 * args.batch_size,
            "learning_rate": 3e-4, "weight_decay": 0.1, "warmup_steps": min(20, max(1, args.steps // 10)),
            "lr_scheduler": "cosine",
            "min_lr_ratio": 0.1, "max_steps": args.steps, "max_grad_norm": 1.0, "log_interval": 5,
            "eval_interval": args.eval_every, "eval_max_batches": 5, "num_workers": 0,
        },
        "distributed": {"strategy": "ddp", "backend": "gloo", "device": "cpu", "timeout_minutes": 10},
        # The two jobs differ only in how often they checkpoint, which must not matter.
        "checkpoint": {"save_interval_steps": save_every, "keep_last_n": 2, "async_save": True},
        "elastic": {"max_restarts": args.max_restarts},
        "fault_tolerance": {"stop_grace_seconds": 10},
    }
    path.write_text(yaml.safe_dump(config))


def job_env(src: Path, home: Path, threads: int) -> Dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("FLEXTRAIN_")}
    env.update(FLEXTRAIN_HOME=str(home), PYTHONPATH=str(src), OMP_NUM_THREADS=str(threads),
               MKL_NUM_THREADS=str(threads), VECLIB_MAXIMUM_THREADS=str(threads))
    return env


def check_code_is_frozen(src: Path, env: Dict[str, str]) -> None:
    out = subprocess.run([sys.executable, "-c", "import flextrain; print(flextrain.__file__)"],
                         cwd=src, env=env, capture_output=True, text=True, check=True).stdout.strip()
    if Path(out).resolve().parent != (src / "flextrain").resolve():
        raise SystemExit(f"the jobs would import flextrain from {out}, not from the frozen copy in {src}")


def launch(src: Path, config: Path, env: Dict[str, str], log_path: Path, max_restarts: int) -> subprocess.Popen:
    cmd = [sys.executable, "-m", "flextrain.cli.main", "launch", "-c", str(config), "--nproc-per-node", "2",
           "--max-restarts", str(max_restarts), str(src / TRAIN_SCRIPT)]
    return subprocess.Popen(cmd, cwd=src, env=env, stdout=open(log_path, "ab"), stderr=subprocess.STDOUT,
                            start_new_session=True)


# ---------------------------------------------------------------------------- observing a job
def process_table() -> List[tuple]:
    out = subprocess.run(["ps", "-axo", "pid=,ppid=,command="], capture_output=True, text=True).stdout
    rows = []
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) == 3:
            rows.append((int(parts[0]), int(parts[1]), parts[2]))
    return rows


def worker_pids(launcher_pid: int) -> List[int]:
    """The training processes of the job: children of the agent, which is the launcher's child."""
    table = process_table()
    agents = {pid for pid, ppid, cmd in table if ppid == launcher_pid and "flextrain.elastic.agent" in cmd}
    return sorted(pid for pid, ppid, cmd in table if ppid in agents and TRAIN_SCRIPT.name in cmd)


def rank_of(pid: int) -> Optional[int]:
    """The worker's rank, from its environment (``ps -E``); ``None`` where that is not readable."""
    out = subprocess.run(["ps", "-Eww", "-p", str(pid), "-o", "command="], capture_output=True, text=True).stdout
    match = re.search(r"(?<![A-Z_])RANK=(\d+)", out)
    return int(match.group(1)) if match else None


def run_record(home: Path) -> Dict[str, Any]:
    for path in (home / "runs").glob("*.json"):
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue  # being replaced right now
    return {}


# ---------------------------------------------------------------------------- comparing results
def differences(a: Any, b: Any, where: str, found: List[str], limit: int = 20) -> None:
    """Append to ``found`` where ``a`` and ``b`` differ; tensors and floats must match bit for bit."""
    if len(found) >= limit:
        return
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        same = (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.dtype == b.dtype
                and a.shape == b.shape and torch.equal(a, b))
        if not same:
            found.append(where)
    elif isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b), key=str):
            if key not in a or key not in b:
                found.append(f"{where}.{key} (only in one)")
            else:
                differences(a[key], b[key], f"{where}.{key}", found, limit)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            found.append(f"{where} (length {len(a)} vs {len(b)})")
        for i, (x, y) in enumerate(zip(a, b, strict=False)):
            differences(x, y, f"{where}[{i}]", found, limit)
    elif a != b:
        found.append(f"{where} ({a!r} vs {b!r})")


def count_tensors(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return 1
    if isinstance(value, dict):
        return sum(count_tensors(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(count_tensors(v) for v in value)
    return 0


def model_digest(state: Dict[str, Any]) -> str:
    sha = hashlib.sha256()
    for name in sorted(state["model"]):
        sha.update(name.encode())
        sha.update(state["model"][name].detach().cpu().contiguous().numpy().tobytes())
    return sha.hexdigest()


def compare_final_checkpoints(reference_out: Path, killed_out: Path, steps: int) -> Dict[str, Any]:
    from flextrain.checkpoint import LocalStorage

    name = f"checkpoint_step{steps:08d}.pt"
    states = {}
    for label, out in (("reference", reference_out), ("killed", killed_out)):
        path = out / EXPERIMENT / "checkpoints" / name
        if not path.exists():
            return {"identical": False, "differences": [f"{label} has no {name}"]}
        states[label] = LocalStorage().load(path)  # also verifies the checkpoint against its digest
    ref, killed = states["reference"], states["killed"]
    found: List[str] = []
    for part in ("model", "optimizer", "lr_scheduler", "grad_scaler", "rng", "epoch", "step", "world_size",
                 "global_batch"):
        differences(ref.get(part), killed.get(part), part, found)
    for part in ("global_step", "epoch", "sampler", "nonfinite_guard"):
        differences(ref["trainer"].get(part), killed["trainer"].get(part), f"trainer.{part}", found)
    tensors = count_tensors(ref.get("model")) + count_tensors(ref.get("optimizer"))
    return {"identical": not found, "differences": found, "tensors_compared": tensors,
            "model_sha256": {"reference": model_digest(ref), "killed": model_digest(killed)},
            "killed_rollbacks": len(killed["trainer"].get("rollback", {}).get("history", []))}


# ---------------------------------------------------------------------------- the run
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, required=True, help="optimizer steps both jobs train for")
    parser.add_argument("--ckpt-every", type=int, required=True, help="checkpoint cadence of the killed job, in steps")
    parser.add_argument("--mean-kill-seconds", type=float, default=480.0, help="mean gap between kills")
    parser.add_argument("--min-kill-gap", type=float, default=90.0, help="shortest gap between kills, in seconds")
    parser.add_argument("--model-size", default="small", choices=["tiny", "small", "medium", "large"])
    parser.add_argument("--batch-size", type=int, default=8, help="per-rank batch size")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--threads", type=int, default=1, help="intra-op threads per rank")
    parser.add_argument("--max-restarts", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dir", default=None, help="working directory (default: a temp dir)")
    parser.add_argument("--keep", action="store_true", help="keep the working directory even if the test passes")
    parser.add_argument("--output", default=None, help="write results JSON here")
    args = parser.parse_args()

    work = Path(args.dir) if args.dir else Path(tempfile.mkdtemp(prefix="flextrain-soak-"))
    work.mkdir(parents=True, exist_ok=True)
    src, code_sha256 = freeze(work)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                            text=True).stdout.strip() or None
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "flextrain", "examples"], cwd=REPO_ROOT,
                                capture_output=True, text=True).stdout.strip())

    jobs: Dict[str, Dict[str, Any]] = {}
    for label, save_every in (("reference", 0), ("killed", args.ckpt_every)):
        root = work / label
        root.mkdir(exist_ok=True)
        write_config(root / "config.yaml", root / "outputs", work / "corpus.txt", args, save_every)
        jobs[label] = {"root": root, "home": root / "home", "out": root / "outputs",
                       "env": job_env(src, root / "home", args.threads)}
    check_code_is_frozen(src, jobs["killed"]["env"])
    log(f"working directory {work}; code frozen at commit {commit}{' plus uncommitted changes' if dirty else ''}")

    rng = random.Random(args.seed)
    kills: List[Dict[str, Any]] = []
    started = time.monotonic()
    try:
        for job in jobs.values():
            job["proc"] = launch(src, job["root"] / "config.yaml", job["env"], job["root"] / "job.log",
                                 args.max_restarts)
            job["started"] = time.monotonic()
        killed = jobs["killed"]
        next_kill = time.monotonic() + max(args.min_kill_gap, rng.expovariate(1 / args.mean_kill_seconds))
        next_report = time.monotonic() + 300
        while any(job["proc"].poll() is None for job in jobs.values()):
            time.sleep(1)
            now = time.monotonic()
            for job in jobs.values():
                if job["proc"].poll() is not None and "seconds" not in job:
                    job["seconds"] = now - job["started"]
            if killed["proc"].poll() is None and now >= next_kill:
                victims = worker_pids(killed["proc"].pid)
                if len(victims) == 2:  # otherwise the job is between incarnations: try again in a second
                    victim = rng.choice(victims)
                    record = run_record(killed["home"])
                    rank = rank_of(victim)
                    try:
                        os.kill(victim, signal.SIGKILL)
                    except ProcessLookupError:
                        continue
                    kills.append({"minutes_in": round((now - started) / 60, 2), "rank": rank,
                                  "step_reported": record.get("step"),
                                  "last_checkpoint_step": record.get("last_checkpoint_step"),
                                  "restarts_before": record.get("restart_count")})
                    log(f"kill {len(kills)}: rank {rank} at about step {record.get('step')} "
                        f"(last checkpoint: step {record.get('last_checkpoint_step')})")
                    next_kill = now + max(args.min_kill_gap, rng.expovariate(1 / args.mean_kill_seconds))
            if now >= next_report:
                next_report = now + 300
                log("progress: " + ", ".join(f"{label} at step {run_record(job['home']).get('step')}/{args.steps}"
                                              for label, job in jobs.items()) + f"; {len(kills)} kills so far")
    except BaseException:
        for job in jobs.values():  # stop both jobs the coordinated way before giving up
            if "proc" in job and job["proc"].poll() is None:
                job["proc"].send_signal(signal.SIGTERM)
        for job in jobs.values():
            if "proc" in job:
                try:
                    job["proc"].wait(timeout=120)
                except subprocess.TimeoutExpired:
                    os.killpg(job["proc"].pid, signal.SIGKILL)
        raise

    summary: Dict[str, Any] = {}
    for label, job in jobs.items():
        record = run_record(job["home"])
        summary[label] = {"exit_code": job["proc"].returncode, "status": record.get("status"),
                          "final_step": record.get("step"), "restarts": record.get("restart_count"),
                          "hours": round(job.get("seconds", time.monotonic() - job["started"]) / 3600, 3)}
    comparison = compare_final_checkpoints(jobs["reference"]["out"], jobs["killed"]["out"], args.steps)
    restarts = summary["killed"]["restarts"]
    passed = (comparison["identical"] and all(s["exit_code"] == 0 and s["status"] == "completed"
                                               and s["final_step"] == args.steps for s in summary.values())
              and summary["reference"]["restarts"] == 0 and restarts == len(kills))

    results = {
        "passed": passed,
        "environment": {"platform": platform.platform(), "python": platform.python_version(),
                        "torch": torch.__version__, "cpu_count": os.cpu_count(), "commit": commit,
                        "uncommitted_changes_in_code": dirty, "code_sha256": code_sha256},
        "settings": {k: v for k, v in vars(args).items() if k not in ("dir", "output", "keep")},
        "jobs": summary, "kills_delivered": len(kills), "kills": kills, "comparison": comparison,
    }
    log(f"reference: {summary['reference']}")
    log(f"killed:    {summary['killed']}")
    log(f"kills delivered: {len(kills)}; final checkpoints identical: {comparison['identical']}"
        + (f"; differences: {comparison['differences']}" if comparison["differences"] else ""))
    log("PASSED" if passed else "FAILED")
    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
        log(f"results written to {args.output}")
    if passed and not args.keep and not args.dir:
        shutil.rmtree(work, ignore_errors=True)
    else:
        log(f"working directory kept: {work}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
