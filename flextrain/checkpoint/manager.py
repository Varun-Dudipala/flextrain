"""Checkpoint manager: naming, saving (sync or async), the commit log, retention, and resume.

Invariants:

* Files are named ``checkpoint_step{step:08d}.pt``. A file in the directory is not yet a
  checkpoint: it counts once it is listed in the directory's manifest (see ``manifest``),
  which is written after the file and its digest are durable.
* The manifest belongs to one *lineage* (one continuous training history). Auto-resume and
  retention only ever look at the current lineage's entries, so leftovers from an earlier run
  in the same directory can neither be resumed by accident nor crowd out new checkpoints, and
  retention never deletes a file this lineage did not write.
* Within a lineage "latest" means highest step, never newest mtime (clock skew, copies and
  re-saves make mtime unreliable). Continuing from step S drops the entries after S - they
  belong to a future that was abandoned - so highest step and most recent commit agree.
* Retention runs as part of the commit, after the new checkpoint is durable: an in-flight or
  failed save can never cause an older good checkpoint to be deleted.
* Resume walks the lineage newest -> oldest and skips checkpoints that are *corrupted*
  (unreadable bytes, or bytes that fail their SHA-256 digest). It does not skip checkpoints it
  merely failed to reach: storage errors are retried and then raised. If the lineage has
  checkpoints but none is usable, that is an error too - starting again from step 0 without
  being asked is how a storage outage turns into lost work.
* One writer per directory: a process that finds another writer's token in the manifest stops
  with ``CheckpointConflictError``.
"""

from __future__ import annotations

import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from flextrain.config import CheckpointConfig

from .async_writer import AsyncCheckpointWriter, CheckpointSaveError, SaveRecord, _save_with_retries
from .manifest import (
    MANIFEST_NAME,
    CheckpointConflictError,
    Manifest,
    ManifestEntry,
    ManifestError,
    NoUsableCheckpointError,
)
from .state import capture_state, restore_state, validate_checkpoint
from .storage import (
    CheckpointCorruptedError,
    LocalStorage,
    StorageBackend,
    UntrustedCheckpointError,
    create_storage_backend,
)

logger = logging.getLogger(__name__)

CHECKPOINT_PATTERN = re.compile(r"^checkpoint_step(\d+)\.pt$")
_STALE_TMP_PATTERN = re.compile(r"^\.(checkpoint_step\d+\.pt(\.sha256)?|manifest\.json)\.[0-9a-f]+\.tmp$")


@dataclass(frozen=True)
class CheckpointInfo:
    path: str
    step: int


class _CommittingStore:
    """What the background writer saves through: persist the bytes, then commit them."""

    def __init__(self, manager: CheckpointManager):
        self._manager = manager

    def save(self, state: Dict[str, Any], path: str) -> int:
        nbytes = self._manager.storage.save(state, path)
        self._manager._commit(path, nbytes)
        return nbytes


class CheckpointManager:
    """Saves, commits, prunes and restores checkpoints. Only rank 0 touches storage on save."""

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
        self.storage = storage or create_storage_backend(
            config.storage_backend, bucket_name=config.bucket_name, checksum=config.checksum,
            allow_pickle=config.allow_pickle)
        self.write_retries = write_retries
        self._store = _CommittingStore(self)

        self._writer: Optional[AsyncCheckpointWriter] = None
        if config.async_save and self.is_main_process:
            self._writer = AsyncCheckpointWriter(
                self._store,
                max_pending=config.max_pending_saves,
                pin_memory=pin_memory,
                write_retries=write_retries,
            )
        self.last_blocking_seconds: Optional[float] = None
        self.last_saved_step: Optional[int] = None
        self.last_committed: Optional[CheckpointInfo] = None  # newest checkpoint this process made durable
        self._sync_history: List[SaveRecord] = []

        # Lineage state. The manifest is read lazily and written by whichever thread commits.
        self._lock = threading.RLock()
        self._writer_id = uuid.uuid4().hex
        self._manifest: Optional[Manifest] = None  # this process's view; None until a lineage is chosen
        self._committed = False  # has this process written the manifest yet?
        self._resumed_step: Optional[int] = None

        if self.is_main_process and isinstance(self.storage, LocalStorage):
            Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
            self._remove_stale_temp_files()
        logger.debug("CheckpointManager(dir=%s, backend=%s, async=%s)",
                     self.checkpoint_dir, config.storage_backend, config.async_save)

    # ------------------------------------------------------------------ naming / listing
    def path_for_step(self, step: int) -> str:
        return self.storage.join(self.checkpoint_dir, f"checkpoint_step{step:08d}.pt")

    @property
    def manifest_path(self) -> str:
        return self.storage.join(self.checkpoint_dir, MANIFEST_NAME)

    def list_checkpoints(self) -> List[CheckpointInfo]:
        """Every checkpoint *file* in ``checkpoint_dir``, oldest step first, whatever wrote it."""
        infos = []
        for path in self.storage.list(self.checkpoint_dir):
            match = CHECKPOINT_PATTERN.match(Path(path).name)
            if match:
                infos.append(CheckpointInfo(path=path, step=int(match.group(1))))
        return sorted(infos, key=lambda info: info.step)

    def stored_manifest(self) -> Optional[Manifest]:
        """The manifest currently on storage, or ``None`` (none written, or unreadable)."""
        raw = self.storage.read_bytes(self.manifest_path)
        if raw is None:
            return None
        try:
            return Manifest.from_bytes(raw)
        except ManifestError as e:
            logger.error("Ignoring %s (%s); falling back to the directory listing", self.manifest_path, e)
            return None

    def _adopted_entries(self) -> List[ManifestEntry]:
        """Entries for a directory written before manifests existed: every checkpoint file in it."""
        return [ManifestEntry(step=info.step, file=Path(info.path).name) for info in self.list_checkpoints()]

    def committed_checkpoints(self) -> List[CheckpointInfo]:
        """The current lineage's committed checkpoints, oldest step first."""
        with self._lock:
            manifest = self._manifest or self.stored_manifest()
            entries = manifest.sorted_entries() if manifest is not None else self._adopted_entries()
        return [CheckpointInfo(path=self.storage.join(self.checkpoint_dir, e.file), step=e.step) for e in entries]

    def latest_checkpoint(self) -> Optional[str]:
        checkpoints = self.committed_checkpoints()
        return checkpoints[-1].path if checkpoints else None

    # ------------------------------------------------------------------ lineage
    def open_lineage(self, origin: Optional[str] = None, launch_id: Optional[str] = None) -> str:
        """Begin a new lineage in this directory (a fresh start, or a branch from ``origin``).

        Nothing is written until the first checkpoint commits. Checkpoints already in the
        directory stay where they are but stop being managed: this lineage will neither resume
        from them nor prune them.
        """
        with self._lock:
            leftovers = len(self.list_checkpoints()) if self.is_main_process else 0
            if leftovers:
                logger.warning("%d checkpoint file(s) from an earlier run remain in %s; this run starts a new "
                               "lineage and will neither resume from nor delete them",
                               leftovers, self.checkpoint_dir)
            self._manifest = Manifest(origin=origin, launch_id=launch_id)
            return self._manifest.lineage

    def continue_lineage(self, step: int, launch_id: Optional[str] = None, lineage: Optional[str] = None) -> str:
        """Continue the directory's lineage from its checkpoint at ``step``.

        Entries after ``step`` are dropped: they were skipped as corrupted, or belong to a future
        this run is no longer on. A directory without a manifest is adopted as it stands.
        """
        with self._lock:
            manifest = self.stored_manifest()
            if manifest is None:
                manifest = Manifest(checkpoints=self._adopted_entries())
                if lineage:
                    manifest.lineage = lineage
            manifest.checkpoints = [e for e in manifest.checkpoints if e.step <= step]
            manifest.launch_id = launch_id
            self._manifest = manifest
            return manifest.lineage

    def _ensure_lineage(self) -> Manifest:
        """Used without an explicit choice (e.g. a bare manager): continue whatever is there."""
        if self._manifest is None:
            manifest = self.stored_manifest() or Manifest(checkpoints=self._adopted_entries())
            if self._resumed_step is not None:
                manifest.checkpoints = [e for e in manifest.checkpoints if e.step <= self._resumed_step]
            self._manifest = manifest
        return self._manifest

    def _commit(self, path: str, nbytes: int) -> None:
        """Make the durable checkpoint at ``path`` part of the lineage, then prune.

        Runs on whichever thread wrote the checkpoint. The manifest is replaced atomically and
        only then are pruned files deleted, so the manifest never names a file that is gone.
        """
        name = Path(path).name
        match = CHECKPOINT_PATTERN.match(name)
        if match is None:
            raise ValueError(f"not a checkpoint file name: {name}")
        step = int(match.group(1))
        with self._lock:
            manifest = self._ensure_lineage()
            stored = self.stored_manifest()
            if self._committed and (stored is None or stored.writer != self._writer_id
                                    or stored.generation != manifest.generation):
                raise CheckpointConflictError(
                    f"another process is writing checkpoints to {self.checkpoint_dir} "
                    f"(the manifest changed since this process last wrote it); refusing to interleave"
                )
            entries = [e for e in manifest.checkpoints if e.step != step]
            entries.append(ManifestEntry(step=step, file=name, nbytes=nbytes, time=time.time()))
            entries.sort(key=lambda e: e.step)
            pruned = entries[: max(0, len(entries) - self.config.keep_last_n)]
            manifest.checkpoints = entries[len(pruned):]
            manifest.writer = self._writer_id
            manifest.generation = (stored.generation if stored is not None else 0) + 1
            self.storage.write_bytes(self.manifest_path, manifest.to_bytes())
            self._committed = True
            if any(e.step == step for e in manifest.checkpoints):
                self.last_committed = CheckpointInfo(path=path, step=step)
        for entry in pruned:
            old = self.storage.join(self.checkpoint_dir, entry.file)
            try:
                self.storage.delete_checkpoint(old)
                logger.debug("Pruned checkpoint %s", old)
            except Exception as e:  # leftover files are harmless; the commit already succeeded
                logger.warning("Failed to delete old checkpoint %s: %s", old, e)

    # ------------------------------------------------------------------ saving
    def save_state(self, state: Dict[str, Any], step: int, blocking: bool = False) -> Optional[str]:
        """Persist an already-captured state dict. Non-main ranks return ``None``.

        With ``async_save`` the call blocks only for the host snapshot; ``blocking=True``
        additionally waits until this (and every earlier) save is durable and committed.
        """
        if not self.is_main_process:
            self.last_saved_step = step  # keep in sync on all ranks: callers branch on it collectively
            return None
        state = dict(state)
        state["step"] = step
        path = self.path_for_step(step)
        start = time.perf_counter()

        if self._writer is not None:
            self._writer.submit(state, path)
            if blocking:
                self._writer.wait()
            self.last_blocking_seconds = time.perf_counter() - start
            logger.info("Checkpoint step %d %s (blocked %.1f ms)", step,
                        "saved" if blocking else "queued", self.last_blocking_seconds * 1e3)
        else:
            nbytes = _save_with_retries(self._store, state, path, self.write_retries)
            self.last_blocking_seconds = time.perf_counter() - start
            self._sync_history.append(SaveRecord(path, nbytes, 0.0, self.last_blocking_seconds))
            logger.info("Checkpoint saved: %s (%.1f MB, blocked %.1f ms)",
                        path, nbytes / 1e6, self.last_blocking_seconds * 1e3)
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

    def _remove_stale_temp_files(self) -> None:
        """Delete temp files left behind by a writer that died mid-save."""
        for path in Path(self.checkpoint_dir).glob(".*.tmp"):
            if _STALE_TMP_PATTERN.match(path.name):
                path.unlink(missing_ok=True)
                logger.info("Removed partial file %s", path)

    # ------------------------------------------------------------------ loading
    def load_state(self, path: str, retries: Optional[int] = None) -> Dict[str, Any]:
        """Load and validate one checkpoint.

        Raises ``CheckpointCorruptedError`` if its bytes are wrong and ``FileNotFoundError`` if
        it is not there. Any other failure is the storage's (a stale handle, a throttled or
        unreachable bucket): it is retried with backoff and then raised as it is.
        """
        retries = self.write_retries if retries is None else retries
        for attempt in range(retries + 1):
            try:
                state = self.storage.load(path)
                break
            except (CheckpointCorruptedError, UntrustedCheckpointError, FileNotFoundError):
                raise  # not transient: retrying cannot help
            except Exception as e:
                if attempt == retries:
                    raise
                delay = 2 ** attempt
                logger.warning("Reading checkpoint %s failed (%s: %s); retrying in %ss",
                               path, type(e).__name__, e, delay)
                time.sleep(delay)
        problems = validate_checkpoint(state)
        if problems:
            raise CheckpointCorruptedError(f"invalid checkpoint {path}: {'; '.join(problems)}")
        return state

    def load_latest_state(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        """The lineage's newest usable checkpoint, skipping corrupted ones.

        Returns ``None`` only when the lineage has no checkpoints at all. Raises
        ``NoUsableCheckpointError`` if it has some and every one is corrupted or missing, and
        lets storage errors propagate - neither case is a reason to start again from scratch.
        """
        candidates = self.committed_checkpoints()
        skipped = []
        for info in reversed(candidates):
            try:
                state = self.load_state(info.path)
            except (CheckpointCorruptedError, FileNotFoundError) as e:
                logger.warning("Skipping unusable checkpoint %s: %s", info.path, e)
                skipped.append(f"step {info.step}: {e}")
                continue
            self._resumed_step = info.step
            return info.path, state
        if candidates:
            raise NoUsableCheckpointError(
                f"all {len(candidates)} checkpoint(s) in {self.checkpoint_dir} are unusable "
                f"({'; '.join(skipped)}). Refusing to silently start from step 0: repair or remove them, "
                f"or set checkpoint.auto_resume=false to begin a new lineage."
            )
        return None

    def load(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
        checkpoint_path: Optional[str] = None,
        strict: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Restore state in place. ``checkpoint_path=None`` means newest usable checkpoint."""
        if checkpoint_path is not None:
            path, state = checkpoint_path, self.load_state(checkpoint_path)
        else:
            found = self.load_latest_state()
            if found is None:
                raise FileNotFoundError(f"No checkpoint found in {self.checkpoint_dir}")
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
        """Like ``load`` but returns ``None`` when the lineage has no checkpoints."""
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
