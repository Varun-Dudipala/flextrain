"""Checkpoint configuration."""

from dataclasses import dataclass
from typing import Optional
from .base import BaseConfig


@dataclass
class CheckpointConfig(BaseConfig):
    """Configuration for checkpoint management."""

    checkpoint_dir: str = "./checkpoints"
    storage_backend: str = "local"  # local | gcs | s3
    bucket_name: Optional[str] = None

    save_interval_steps: int = 500
    save_interval_minutes: Optional[int] = None

    # Async I/O
    async_save: bool = True
    num_io_workers: int = 4

    # Versioning
    keep_last_n: int = 5
    keep_best_n: int = 3
    best_metric: str = "loss"
    best_mode: str = "min"

    # FSDP
    save_full_state: bool = False
    save_optimizer: bool = True
    save_rng_state: bool = True

    # Resume
    auto_resume: bool = True
    resume_path: Optional[str] = None
    strict_resume: bool = True

    emergency_checkpoint: bool = True

    def validate(self) -> None:
        valid_backends = {"local", "gcs", "s3"}
        if self.storage_backend.lower() not in valid_backends:
            raise ValueError(f"storage_backend must be one of {valid_backends}")
        if self.keep_last_n < 1:
            raise ValueError("keep_last_n must be at least 1")
