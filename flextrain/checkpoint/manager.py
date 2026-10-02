"""Checkpoint manager: naming, saving (sync or async), retention, and resume.

Invariants:

* Files are named ``checkpoint_step{step:08d}.pt``; "latest" means highest step,
  never newest mtime (clock skew, copies and re-saves make mtime unreliable).
* Retention is computed from the storage listing, so it keeps working across job
  restarts, and it runs only after a write is durable - an in-flight or failed save
  can never cause an older good checkpoint to be deleted.
* Resume walks checkpoints newest -> oldest and skips unreadable ones, so a corrupted
  latest checkpoint costs a few steps of progress instead of the whole job.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from flextrain.config import CheckpointConfig

from .async_writer import AsyncCheckpointWriter, CheckpointSaveError, SaveRecord, _save_with_retries
from .state import capture_state, restore_state, validate_checkpoint
from .storage import LocalStorage, StorageBackend, create_storage_backend

logger = logging.getLogger(__name__)

CHECKPOINT_PATTERN = re.compile(r"^checkpoint_step(\d+)\.pt$")
_STALE_TMP_PATTERN = re.compile(r"^\.checkpoint_step\d+\.pt\.[0-9a-f]+\.tmp$")


@dataclass(frozen=True)
class CheckpointInfo:
    path: str
    step: int


class CheckpointManager:
    """Saves, prunes and restores checkpoints. Only rank 0 touches storage on save."""

    def __init__(
        self,
        config: CheckpointConfig,
        rank: int = 0,
        world_size: int = 1,
        checkpoint_dir: Optional[str] = None,
        storage: Optional[StorageBackend] = None,
        pin_memory: bool = False,
        write_retries: int = 2,
    ):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.is_main_process = rank == 0
        self.checkpoint_dir = checkpoint_dir or config.checkpoint_dir or "./checkpoints"
        self.storage = storage or create_storage_backend(config.storage_backend, bucket_name=config.bucket_name)
        self.write_retries = write_retries

        self._writer: Optional[AsyncCheckpointWriter] = None
        if config.async_save and self.is_main_process:
            self._writer = AsyncCheckpointWriter(
                self.storage,
                max_pending=config.max_pending_saves,
                pin_memory=pin_memory,
                write_retries=write_retries,
            )
        self.last_blocking_seconds: Optional[float] = None
        self.last_saved_step: Optional[int] = None
        self._sync_history: List[SaveRecord] = []

        if self.is_main_process and isinstance(self.storage, LocalStorage):
            Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
            self._remove_stale_temp_files()
        logger.debug("CheckpointManager(dir=%s, backend=%s, async=%s)",
                     self.checkpoint_dir, config.storage_backend, config.async_save)

    # ------------------------------------------------------------------ naming / listing
    def path_for_step(self, step: int) -> str:
        return self.storage.join(self.checkpoint_dir, f"checkpoint_step{step:08d}.pt")

    def list_checkpoints(self) -> List[CheckpointInfo]:
        """All checkpoints in ``checkpoint_dir``, oldest step first."""
        infos = []
        for path in self.storage.list(self.checkpoint_dir):
            match = CHECKPOINT_PATTERN.match(Path(path).name)
            if match:
                infos.append(CheckpointInfo(path=path, step=int(match.group(1))))
        return sorted(infos, key=lambda info: info.step)

    def latest_checkpoint(self) -> Optional[str]:
        checkpoints = self.list_checkpoints()
        return checkpoints[-1].path if checkpoints else None

    # ------------------------------------------------------------------ saving
    def save_state(self, state: Dict[str, Any], step: int, blocking: bool = False) -> Optional[str]:
        """Persist an already-captured state dict. Non-main ranks return ``None``.

        With ``async_save`` the call blocks only for the host snapshot; ``blocking=True``
        additionally waits until this (and every earlier) save is durable.
        """
        if not self.is_main_process:
            self.last_saved_step = step  # keep in sync on all ranks: callers branch on it collectively
            return None
        state = dict(state)
        state["step"] = step
        path = self.path_for_step(step)
        start = time.perf_counter()

        if self._writer is not None:
            self._writer.submit(state, path, on_success=self._on_saved)
            if blocking:
                self._writer.wait()
            self.last_blocking_seconds = time.perf_counter() - start
            logger.info("Checkpoint step %d %s (blocked %.1f ms)", step,
                        "saved" if blocking else "queued", self.last_blocking_seconds * 1e3)
        else:
            nbytes = _save_with_retries(self.storage, state, path, self.write_retries)
            self.last_blocking_seconds = time.perf_counter() - start
            self._sync_history.append(SaveRecord(path, nbytes, 0.0, self.last_blocking_seconds))
            logger.info("Checkpoint saved: %s (%.1f MB, blocked %.1f ms)",
                        path, nbytes / 1e6, self.last_blocking_seconds * 1e3)
            self._on_saved(path)
        self.last_saved_step = step
        return path

    def save(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
        step: int = 0,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        extra: Optional[Dict[str, Any]] = None,
        blocking: bool = False,
    ) -> Optional[str]:
        """Capture and save model/optimizer/scheduler state.

        Must be called on **every** rank (gathering FSDP state is a collective).
        """
        if not self.config.save_optimizer:
            optimizer = None
        state = capture_state(model, optimizer, lr_scheduler)
        state.update(epoch=epoch, metrics=dict(metrics or {}), world_size=self.world_size)
        if extra:
            state.update(extra)
        return self.save_state(state, step=step, blocking=blocking)

    def _on_saved(self, path: str) -> None:
        self._prune()

    def _prune(self) -> None:
        checkpoints = self.list_checkpoints()
        for info in checkpoints[: max(0, len(checkpoints) - self.config.keep_last_n)]:
            try:
                self.storage.delete(info.path)
                logger.debug("Pruned checkpoint %s", info.path)
            except Exception as e:
                logger.warning("Failed to delete old checkpoint %s: %s", info.path, e)

    def _remove_stale_temp_files(self) -> None:
        """Delete temp files left behind by a writer that died mid-save."""
        for path in Path(self.checkpoint_dir).glob(".checkpoint_step*.tmp"):
            if _STALE_TMP_PATTERN.match(path.name):
                path.unlink(missing_ok=True)
                logger.info("Removed partial checkpoint %s", path)

    # ------------------------------------------------------------------ loading
    def load_state(self, path: str) -> Dict[str, Any]:
        """Load and validate a single checkpoint (raises if unusable)."""
        state = self.storage.load(path)
        problems = validate_checkpoint(state)
        if problems:
            raise ValueError(f"invalid checkpoint {path}: {'; '.join(problems)}")
        return state

    def load_latest_state(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Newest checkpoint that loads and validates, skipping corrupted ones."""
        for info in reversed(self.list_checkpoints()):
            try:
                return info.path, self.load_state(info.path)
            except Exception as e:
                logger.warning("Skipping unusable checkpoint %s: %s", info.path, e)
        return None

    def load(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
        checkpoint_path: Optional[str] = None,
        strict: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Restore state in place. ``checkpoint_path=None`` means newest valid checkpoint."""
        if checkpoint_path is not None:
            path, state = checkpoint_path, self.load_state(checkpoint_path)
        else:
            found = self.load_latest_state()
            if found is None:
                raise FileNotFoundError(f"No usable checkpoint found in {self.checkpoint_dir}")
            path, state = found
        restore_state(state, model, optimizer, lr_scheduler,
                      strict=self.config.strict_resume if strict is None else strict)
        logger.info("Restored checkpoint %s (step %d)", path, state["step"])
        return {"path": path, "step": state["step"], "epoch": state.get("epoch", 0),
                "metrics": state.get("metrics", {})}

    def try_load_latest(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        """Like ``load`` but returns ``None`` when there is nothing to resume from."""
        try:
            return self.load(model, optimizer, lr_scheduler)
        except FileNotFoundError:
            return None

    # ------------------------------------------------------------------ lifecycle
    def wait_for_pending(self) -> None:
        """Block until queued async saves are durable; re-raises a failed save."""
        if self._writer is not None:
            pending = self._writer.pending_count()
            if pending:
                logger.info("Waiting for %d pending checkpoint write(s)...", pending)
            self._writer.wait()

    def pending_saves(self) -> int:
        return self._writer.pending_count() if self._writer is not None else 0

    @property
    def history(self) -> List[SaveRecord]:
        return list(self._writer.history) if self._writer is not None else list(self._sync_history)

    def close(self, raise_errors: bool = True) -> None:
        """Drain pending writes and release host buffers."""
        if self._writer is not None:
            self._writer.close(raise_errors=raise_errors)

    def __enter__(self) -> CheckpointManager:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close(raise_errors=exc_type is None)


__all__ = ["CheckpointManager", "CheckpointInfo", "CheckpointSaveError", "CHECKPOINT_PATTERN"]
