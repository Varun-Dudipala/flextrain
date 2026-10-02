#!/usr/bin/env python3
"""FlexTrain checkpoint benchmark.

Measures, for the example GPT-2 model with a realistic checkpoint (weights + AdamW state):

1. **Training stall per checkpoint**: how long the training loop is blocked by a
   synchronous save vs. an async save (device->host snapshot only). Median of N repeats;
   the first async save (which allocates the reusable host buffers) is reported separately.
2. **Persist throughput**: checkpoint bytes / background serialize+write+fsync time.
3. **Load throughput / resume time**: reading the checkpoint with a *cold* page cache
   (evicted via posix_fadvise) and warm, plus the full restore into model + optimizer.
4. **End-to-end overhead**: wall time of the real ``Trainer`` loop with checkpoints every K
   steps (sync vs. async) relative to the same run without checkpoints.

Usage:
    python scripts/benchmark.py                       # GPT-2 small, auto device
    python scripts/benchmark.py --model-size tiny --steps 40 --ckpt-every 10
    python scripts/benchmark.py --output results.json --dir /mnt/fast-disk/tmp
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "examples" / "gpt2_training"))

from model import create_gpt2_model  # noqa: E402

from flextrain import Config, Trainer  # noqa: E402
from flextrain.checkpoint import CheckpointManager, LocalStorage, restore_state  # noqa: E402
from flextrain.config import CheckpointConfig  # noqa: E402


class RandomTokens(Dataset):
    def __init__(self, n: int, seq_len: int, vocab: int):
        self.n, self.seq_len, self.vocab = n, seq_len, vocab

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        g = torch.Generator().manual_seed(i)
        ids = torch.randint(0, self.vocab, (self.seq_len,), generator=g)
        return {"input_ids": ids, "labels": ids}


def drop_page_cache(path: Path) -> bool:
    """Evict a file from the OS page cache so the next read hits the disk (Linux)."""
    if not hasattr(os, "posix_fadvise"):
        return False
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    return True


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def median(xs: List[float]) -> float:
    return statistics.median(xs) if xs else float("nan")


def make_model_and_optimizer(size: str, device: torch.device, seq_len: int):
    torch.manual_seed(0)
    model = create_gpt2_model(size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    ids = torch.randint(0, model.config.vocab_size, (1, min(seq_len, 64)), device=device)
    model(ids, labels=ids)[1].backward()  # populate AdamW state so the checkpoint is realistic
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return model, optimizer


def bench_checkpoint(model, optimizer, device, workdir: Path, repeats: int) -> Dict[str, float]:
    results: Dict[str, float] = {}

    # --- synchronous: the loop is blocked for capture + serialize + write + fsync
    sync_dir = workdir / "sync"
    mgr = CheckpointManager(CheckpointConfig(checkpoint_dir=str(sync_dir), async_save=False, keep_last_n=2))
    sync_times = []
    for i in range(repeats):
        sync_device(device)
        t0 = time.perf_counter()
        path = mgr.save(model, optimizer, step=i)
        sync_times.append(time.perf_counter() - t0)
    nbytes = Path(path).stat().st_size
    results["checkpoint_mb"] = nbytes / 1e6
    results["sync_block_ms"] = median(sync_times) * 1e3
    mgr.close()

    # --- async: the loop is blocked only for the device->host snapshot
    async_dir = workdir / "async"
    mgr = CheckpointManager(CheckpointConfig(checkpoint_dir=str(async_dir), async_save=True, keep_last_n=2),
                            pin_memory=device.type == "cuda")
    first_block, block_times, write_times = None, [], []
    for i in range(repeats + 1):
        sync_device(device)
        t0 = time.perf_counter()
        mgr.save(model, optimizer, step=i)
        blocked = time.perf_counter() - t0
        mgr.wait_for_pending()
        write_times.append(mgr.history[-1].write_seconds)
        if i == 0:
            first_block = blocked  # includes one-time host buffer allocation
        else:
            block_times.append(blocked)
    mgr.close()
    results["async_block_first_ms"] = first_block * 1e3
    results["async_block_ms"] = median(block_times) * 1e3
    results["persist_mbps"] = nbytes / 1e6 / median(write_times)
    results["stall_reduction_pct"] = 100 * (1 - results["async_block_ms"] / results["sync_block_ms"])

    # --- load / resume
    latest = Path(mgr.latest_checkpoint())
    storage = LocalStorage()
    cold, warm, resume = [], [], []
    for _ in range(repeats):
        evicted = drop_page_cache(latest)
        t0 = time.perf_counter()
        state = storage.load(latest)
        (cold if evicted else warm).append(time.perf_counter() - t0)
        del state
        t0 = time.perf_counter()
        state = storage.load(latest)
        warm.append(time.perf_counter() - t0)
        del state
        drop_page_cache(latest)
        t0 = time.perf_counter()
        restore_state(storage.load(latest), model, optimizer)
        sync_device(device)
        resume.append(time.perf_counter() - t0)
    if cold:
        results["load_cold_mbps"] = nbytes / 1e6 / median(cold)
    results["load_warm_mbps"] = nbytes / 1e6 / median(warm)
    results["resume_s"] = median(resume)
    return results


def bench_training_overhead(size, device, workdir: Path, steps: int, every: int, batch: int, seq_len: int):
    results: Dict[str, float] = {}
    for mode in ("none", "sync", "async"):
        torch.manual_seed(0)
        model = create_gpt2_model(size, max_position_embeddings=max(seq_len, 256))
        cfg = Config()
        cfg.main.output_dir = str(workdir / f"e2e-{mode}")
        cfg.main.log_to_file = False
        cfg.main.log_level = "WARNING"
        cfg.training.batch_size = batch
        cfg.training.max_steps = steps
        cfg.training.num_workers = 0
        cfg.training.log_interval = 10**9
        cfg.checkpoint.save_interval_steps = every if mode != "none" else 0
        cfg.checkpoint.save_final = False
        cfg.checkpoint.async_save = mode == "async"
        cfg.checkpoint.keep_last_n = 1
        cfg.fault_tolerance.handle_signals = False
        trainer = Trainer(model, cfg, RandomTokens(batch * steps, seq_len, model.config.vocab_size))
        # warm-up step outside the timed region (allocator, kernels, autotuning)
        trainer._forward_backward(next(iter(trainer.train_loader)), sync_gradients=True)
        trainer.optimizer.zero_grad(set_to_none=True)
        sync_device(device)
        t0 = time.perf_counter()
        trainer.train()  # includes waiting for the last async write to finish
        sync_device(device)
        results[f"train_{mode}_s"] = time.perf_counter() - t0
        trainer.close()
        del trainer, model
    base = results["train_none_s"]
    results["sync_overhead_pct"] = 100 * (results["train_sync_s"] - base) / base
    results["async_overhead_pct"] = 100 * (results["train_async_s"] - base) / base
    if results["sync_overhead_pct"] > 0:
        results["overhead_reduction_pct"] = 100 * (1 - results["async_overhead_pct"] / results["sync_overhead_pct"])
    results["step_s"] = base / steps
    return results


def environment(device: torch.device, workdir: Path) -> Dict[str, str]:
    env = {
        "torch": torch.__version__,
        "python": platform.python_version(),
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else f"CPU ({os.cpu_count()} cores)",
        "torch_threads": str(torch.get_num_threads()),
        "platform": platform.platform(),
        "workdir": str(workdir),
    }
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-size", default="small", choices=["tiny", "small", "medium", "large"])
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--steps", type=int, default=30, help="training steps for the end-to-end benchmark")
    parser.add_argument("--ckpt-every", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=None, help="default: 8 on GPU, 2 on CPU")
    parser.add_argument("--seq-len", type=int, default=None, help="default: 512 on GPU, 128 on CPU")
    parser.add_argument("--threads", type=int, default=None,
                        help="torch intra-op threads (CPU training: leave a core free for the checkpoint writer)")
    parser.add_argument("--skip-e2e", action="store_true")
    parser.add_argument("--dir", default=None, help="where to write checkpoints (default: a temp dir)")
    parser.add_argument("--output", default=None, help="write results JSON here")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if args.threads:
        torch.set_num_threads(args.threads)
    batch = args.batch_size or (8 if device.type == "cuda" else 2)
    seq_len = args.seq_len or (512 if device.type == "cuda" else 128)
    workdir = Path(tempfile.mkdtemp(prefix="flextrain-bench-", dir=args.dir))
    try:
        model, optimizer = make_model_and_optimizer(args.model_size, device, seq_len)
        params = sum(p.numel() for p in model.parameters())
        print(f"GPT-2 {args.model_size}: {params / 1e6:.1f}M params on {device}; checkpoints in {workdir}")

        ckpt = bench_checkpoint(model, optimizer, device, workdir, args.repeats)
        del model, optimizer
        e2e = {} if args.skip_e2e else bench_training_overhead(
            args.model_size, device, workdir, args.steps, args.ckpt_every, batch, seq_len)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    env = environment(device, workdir)
    print(f"\nEnvironment: {env['device']}, torch {env['torch']}, {env['platform']}")
    print(f"\n| Checkpoint ({ckpt['checkpoint_mb']:.0f} MB: weights + AdamW state) | Result |")
    print("|---|---|")
    print(f"| Training stall, synchronous save | {ckpt['sync_block_ms']:.0f} ms |")
    print(f"| Training stall, async save (steady state) | {ckpt['async_block_ms']:.0f} ms |")
    print(f"| Training stall, async save (first, allocates host buffers) | {ckpt['async_block_first_ms']:.0f} ms |")
    print(f"| **Stall reduction (async vs sync)** | **{ckpt['stall_reduction_pct']:.1f}%** |")
    print(f"| Background persist throughput (serialize + write + fsync) | {ckpt['persist_mbps']:.0f} MB/s |")
    if "load_cold_mbps" in ckpt:
        print(f"| Load throughput, cold page cache | {ckpt['load_cold_mbps']:.0f} MB/s |")
    print(f"| Load throughput, warm page cache | {ckpt['load_warm_mbps']:.0f} MB/s |")
    print(f"| Resume (load + restore model & optimizer) | {ckpt['resume_s']:.2f} s |")
    if e2e:
        print(f"\n| End-to-end: {args.steps} steps, checkpoint every {args.ckpt_every} "
              f"(batch {batch} x {seq_len} tokens) | Result |")
        print("|---|---|")
        print(f"| Step time without checkpoints | {e2e['step_s'] * 1e3:.0f} ms |")
        print(f"| Checkpoint overhead, synchronous | {e2e['sync_overhead_pct']:.1f}% |")
        print(f"| Checkpoint overhead, async | {e2e['async_overhead_pct']:.1f}% |")
        if "overhead_reduction_pct" in e2e:
            print(f"| **Overhead reduction (async vs sync)** | **{e2e['overhead_reduction_pct']:.1f}%** |")

    if args.output:
        out = {"environment": env, "model_size": args.model_size, "params_millions": params / 1e6,
               "checkpoint": ckpt, "end_to_end": e2e, "settings": vars(args)}
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(f"\nResults written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
