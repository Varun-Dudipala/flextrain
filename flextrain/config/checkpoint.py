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

    # Integrity: write a SHA-256 digest next to every checkpoint and verify it on load, so a
    # checkpoint whose bytes changed on storage is skipped instead of silently resumed from.
    checksum: bool = True
    # Loading reconstructs tensors and plain containers only, never arbitrary pickled objects
    # (unpickling can execute code, and a checkpoint location is often writable by others). Set
    # this to load checkpoints that need full unpickling: those written by FlexTrain <= 0.2, or
    # holding custom scheduler / optimizer state. Only do so for files you trust.
    allow_pickle: bool = False

    # Resume
    auto_resume: bool = True  # resume from the newest valid checkpoint in checkpoint_dir
    # Checkpoint to branch a *new* lineage from (e.g. to roll back past a loss spike). It is the
    # starting point, not a standing order: once that lineage has checkpoints of its own, a
    # restart continues from the newest of them instead of going back to resume_path.
    resume_path: Optional[str] = None
    strict_resume: bool = True  # strict model state-dict key matching
    # Continuing a lineage with different optimization settings (learning rate, schedule, seed,
    # model / data sections, ...) than its checkpoint was written with is refused unless this is
    # set: the settings in effect would otherwise be a silent mix of old state and new config.
    allow_config_change: bool = False

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
