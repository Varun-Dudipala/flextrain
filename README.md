# FlexTrain

A fault-tolerant distributed training framework with async checkpointing, elastic scaling, and experiment tracking.

## Features

- **Distributed Training**: DDP and FSDP support with automatic model wrapping
- **Async Checkpointing**: Non-blocking checkpoint saves with background I/O
- **Fault Tolerance**: Automatic failure detection and recovery
- **Elastic Scaling**: Dynamic cluster reconfiguration without restart
- **Experiment Tracking**: Built-in metrics logging with SQLite backend
- **Cloud Storage**: Support for local, GCS, and S3 checkpoint storage
- **CLI & Dashboard**: Command-line tools and web dashboard

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from flextrain import Config, DistributedTrainer

# Create config
config = Config()
config.training.batch_size = 32
config.training.learning_rate = 1e-4

# Create trainer
trainer = DistributedTrainer(
    model=model,
    config=config,
    train_dataset=dataset,
)

# Train with automatic checkpointing and fault tolerance
trainer.train()
```

## Configuration

```yaml
main:
  experiment_name: my_experiment
  output_dir: ./outputs

training:
  batch_size: 32
  learning_rate: 1e-4
  max_steps: 10000

distributed:
  strategy: fsdp
  mixed_precision: true

checkpoint:
  async_save: true
  keep_last_n: 5
```

## Performance Benchmarks

Measured on Tesla T4 GPU (Google Colab):

| Metric | Result |
|--------|--------|
| Resume time (GPT-2 124M) | **3.1 seconds** |
| Checkpoint load throughput | **1,169 MB/s** |
| Checkpoint save throughput | **127.7 MB/s** |
| Async checkpoint blocking | **602ms** (500MB model) |
| Training throughput | **3,690 tokens/sec** |

### Benchmark Details

```
Hardware: Tesla T4 (15GB VRAM)
Model: GPT-2 Small (124M parameters)
Batch size: 8 x 512 tokens
PyTorch: 2.9.0+cu126
```

## CLI Usage

```bash
# Start training
flextrain train config.yaml

# Validate config
flextrain validate config.yaml

# Start dashboard
flextrain serve --port 8000
```

## Project Structure

```
flextrain/
├── config/          # Configuration management
├── core/            # Trainer, DDP/FSDP wrappers
├── checkpoint/      # Checkpoint manager, storage backends
├── fault_tolerance/ # Failure detection and recovery
├── elastic/         # Elastic scaling
├── tracking/        # Experiment tracking
├── cli/             # Command-line interface
└── api/             # REST API and dashboard
```

## License

MIT
