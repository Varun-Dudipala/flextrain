"""Checkpoint configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .base import BaseConfig, _one_of, _require


@dataclass
class CheckpointConfig(BaseConfig):
    """Checkpoint saving, retention and resume settings."""

    # None -> "<main.output_dir>/<main.experiment_name>/checkpoints". Keeping checkpoints
    # per experiment prevents one experiment from auto-resuming another's state.
    # For gcs/s3 this is the object-key prefix inside ``bucket_name``.
    checkpoint_dir: Optional[str] = None
    storage_backend: str = "local"  # local | gcs | s3
    bucket_name: Optional[str] = None

    # When to save (either trigger fires a save; 0 / None disables that trigger)
    save_interval_steps: int = 500
    save_interval_minutes: Optional[float] = None
    save_final: bool = True

    # Async I/O: the training thread only snapshots tensors to host memory; serialization
    # and the write happen on a background thread. ``max_pending_saves`` bounds how many
    # snapshots may be in flight (and therefore host memory); further saves block.
    async_save: bool = True
    max_pending_saves: int = 1

    # Retention: newest ``keep_last_n`` checkpoints (by step) are kept, older ones deleted
    keep_last_n: int = 3

    save_optimizer: bool = True

    # Resume
    auto_resume: bool = True  # resume from the newest valid checkpoint in checkpoint_dir
    resume_path: Optional[str] = None  # explicit checkpoint to resume from (overrides auto_resume)
    strict_resume: bool = True  # strict model state-dict key matching

    @classmethod
    def section_name(cls) -> str:
        return "checkpoint"

    def validate(self) -> None:
        self.storage_backend = self.storage_backend.lower()
        _one_of("checkpoint.storage_backend", self.storage_backend, {"local", "gcs", "s3"})
        if self.storage_backend != "local":
            _require(bool(self.bucket_name),
                     f"checkpoint.bucket_name is required for storage_backend={self.storage_backend!r}")
        _require(self.save_interval_steps >= 0, "checkpoint.save_interval_steps must be >= 0")
        if self.save_interval_minutes is not None:
            _require(self.save_interval_minutes > 0, "checkpoint.save_interval_minutes must be positive")
        _require(self.max_pending_saves >= 1, "checkpoint.max_pending_saves must be >= 1")
        _require(self.keep_last_n >= 1, "checkpoint.keep_last_n must be at least 1")
