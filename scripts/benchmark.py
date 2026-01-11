#!/usr/bin/env python3
"""
FlexTrain Benchmark Script

Measures checkpoint save/load performance, training throughput, and async I/O overhead.
Run this to get reproducible benchmark results.

Usage:
    python scripts/benchmark.py [--device cuda|cpu] [--model-size small|medium]
"""

import argparse
import time
import tempfile
import json
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import torch
import torch.nn as nn

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from flextrain.config import CheckpointConfig
from flextrain.checkpoint import CheckpointManager
from flextrain.checkpoint.storage import LocalStorage


@dataclass
class BenchmarkResults:
    """Benchmark results container."""
    device: str
    model_size: str
    model_params_millions: float
    checkpoint_size_mb: float

    # Checkpoint benchmarks
    sync_save_time_ms: float
    sync_load_time_ms: float
    async_queue_time_ms: float
    async_total_time_ms: float

    # Throughput
    save_throughput_mb_s: float
    load_throughput_mb_s: float

    # Training simulation
    tokens_per_second: float
    training_step_ms: float

    # Overhead
    async_blocking_overhead_percent: float


class BenchmarkModel(nn.Module):
    """Simple transformer-like model for benchmarking."""

    def __init__(self, hidden_size: int = 768, num_layers: int = 12, vocab_size: int = 50257):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=12,
                dim_feedforward=hidden_size * 4,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_model_config(size: str) -> dict:
    """Get model configuration by size."""
    configs = {
        "small": {"hidden_size": 768, "num_layers": 12},   # ~124M params
        "medium": {"hidden_size": 1024, "num_layers": 24}, # ~350M params
        "large": {"hidden_size": 1280, "num_layers": 36},  # ~774M params
    }
    return configs.get(size, configs["small"])


def benchmark_checkpoint_sync(model: nn.Module, tmpdir: str, device: str) -> dict:
    """Benchmark synchronous checkpoint save/load."""
    config = CheckpointConfig(
        checkpoint_dir=tmpdir,
        async_save=False,
        storage_backend="local"
    )
    manager = CheckpointManager(config, rank=0, world_size=1)

    # Measure save time
    start = time.perf_counter()
    path = manager.save(model, step=0, blocking=True)
    save_time = (time.perf_counter() - start) * 1000

    # Get checkpoint size
    ckpt_size_mb = Path(path).stat().st_size / (1024 * 1024)

    # Measure load time
    start = time.perf_counter()
    manager.load(model, checkpoint_path=path)
    load_time = (time.perf_counter() - start) * 1000

    return {
        "save_time_ms": save_time,
        "load_time_ms": load_time,
        "size_mb": ckpt_size_mb,
    }


def benchmark_checkpoint_async(model: nn.Module, tmpdir: str, device: str) -> dict:
    """Benchmark asynchronous checkpoint save."""
    config = CheckpointConfig(
        checkpoint_dir=tmpdir,
        async_save=True,
        num_io_workers=2,
        storage_backend="local"
    )
    manager = CheckpointManager(config, rank=0, world_size=1)

    # Measure queue time (blocking portion)
    start = time.perf_counter()
    path = manager.save(model, step=100, blocking=False)
    queue_time = (time.perf_counter() - start) * 1000

    # Wait for async completion
    async_start = time.perf_counter()
    manager.wait_for_pending()
    total_time = (time.perf_counter() - start) * 1000

    return {
        "queue_time_ms": queue_time,
        "total_time_ms": total_time,
    }


def benchmark_training_step(model: nn.Module, device: str, batch_size: int = 8, seq_len: int = 512) -> dict:
    """Benchmark a single training step."""
    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Warmup
    for _ in range(3):
        input_ids = torch.randint(0, 50257, (batch_size, seq_len), device=device)
        logits = model(input_ids)
        loss = logits.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if device == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    num_steps = 10
    start = time.perf_counter()

    for _ in range(num_steps):
        input_ids = torch.randint(0, 50257, (batch_size, seq_len), device=device)
        logits = model(input_ids)
        loss = logits.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if device == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    step_time_ms = (elapsed / num_steps) * 1000
    tokens_per_step = batch_size * seq_len
    tokens_per_second = tokens_per_step / (elapsed / num_steps)

    return {
        "step_time_ms": step_time_ms,
        "tokens_per_second": tokens_per_second,
    }


def run_benchmarks(device: str, model_size: str) -> BenchmarkResults:
    """Run all benchmarks."""
    print(f"\n{'='*60}")
    print(f"FlexTrain Benchmark")
    print(f"Device: {device}, Model Size: {model_size}")
    print(f"{'='*60}\n")

    # Create model
    model_config = get_model_config(model_size)
    model = BenchmarkModel(**model_config)
    model = model.to(device)

    param_count = model.count_parameters()
    param_millions = param_count / 1e6
    print(f"Model parameters: {param_millions:.1f}M")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Sync checkpoint benchmark
        print("\nBenchmarking synchronous checkpoint...")
        sync_results = benchmark_checkpoint_sync(model, tmpdir, device)
        print(f"  Save time: {sync_results['save_time_ms']:.1f}ms")
        print(f"  Load time: {sync_results['load_time_ms']:.1f}ms")
        print(f"  Size: {sync_results['size_mb']:.1f}MB")

        # Async checkpoint benchmark
        print("\nBenchmarking asynchronous checkpoint...")
        async_results = benchmark_checkpoint_async(model, tmpdir, device)
        print(f"  Queue time (blocking): {async_results['queue_time_ms']:.1f}ms")
        print(f"  Total time: {async_results['total_time_ms']:.1f}ms")

    # Calculate throughput
    save_throughput = sync_results['size_mb'] / (sync_results['save_time_ms'] / 1000)
    load_throughput = sync_results['size_mb'] / (sync_results['load_time_ms'] / 1000)
    print(f"\nThroughput:")
    print(f"  Save: {save_throughput:.1f} MB/s")
    print(f"  Load: {load_throughput:.1f} MB/s")

    # Training benchmark
    print("\nBenchmarking training step...")
    train_results = benchmark_training_step(model, device)
    print(f"  Step time: {train_results['step_time_ms']:.1f}ms")
    print(f"  Tokens/sec: {train_results['tokens_per_second']:.0f}")

    # Calculate async overhead
    async_overhead = (async_results['queue_time_ms'] / train_results['step_time_ms']) * 100
    print(f"\nAsync checkpoint overhead: {async_overhead:.1f}% of step time")

    results = BenchmarkResults(
        device=device,
        model_size=model_size,
        model_params_millions=param_millions,
        checkpoint_size_mb=sync_results['size_mb'],
        sync_save_time_ms=sync_results['save_time_ms'],
        sync_load_time_ms=sync_results['load_time_ms'],
        async_queue_time_ms=async_results['queue_time_ms'],
        async_total_time_ms=async_results['total_time_ms'],
        save_throughput_mb_s=save_throughput,
        load_throughput_mb_s=load_throughput,
        tokens_per_second=train_results['tokens_per_second'],
        training_step_ms=train_results['step_time_ms'],
        async_blocking_overhead_percent=async_overhead,
    )

    return results


def print_summary(results: BenchmarkResults):
    """Print benchmark summary."""
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"""
Device:                     {results.device}
Model Size:                 {results.model_size} ({results.model_params_millions:.1f}M params)
Checkpoint Size:            {results.checkpoint_size_mb:.1f} MB

Checkpoint Performance:
  Sync Save:                {results.sync_save_time_ms:.1f}ms
  Sync Load:                {results.sync_load_time_ms:.1f}ms
  Async Queue (blocking):   {results.async_queue_time_ms:.1f}ms
  Async Total:              {results.async_total_time_ms:.1f}ms

Throughput:
  Save:                     {results.save_throughput_mb_s:.1f} MB/s
  Load:                     {results.load_throughput_mb_s:.1f} MB/s

Training:
  Step Time:                {results.training_step_ms:.1f}ms
  Tokens/sec:               {results.tokens_per_second:.0f}

Async I/O Overhead:         {results.async_blocking_overhead_percent:.1f}% of step time
""")


def main():
    parser = argparse.ArgumentParser(description="FlexTrain Benchmark")
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None,
                        help="Device to use (default: auto-detect)")
    parser.add_argument("--model-size", choices=["small", "medium", "large"], default="small",
                        help="Model size (default: small)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for results")
    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    # Run benchmarks
    results = run_benchmarks(device, args.model_size)
    print_summary(results)

    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(asdict(results), f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
