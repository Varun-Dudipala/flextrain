"""Background checkpoint writer.

Design (and why):

* **One writer thread, FIFO.** Checkpoints are written strictly in submission order, so
  "newest checkpoint on storage" always means "newest step", and retention can run on
  the writer thread right after a successful write without racing other writes.
  Parallel writers to the same disk rarely help throughput and break that ordering.
* **Bounded in-flight saves (backpressure).** At most ``max_pending`` snapshots exist at
  once. If training checkpoints faster than storage can absorb, ``submit`` blocks instead
  of letting host memory grow without bound.
* **Failures surface.** A failed write is re-raised on the training thread at the next
  ``submit`` / ``wait`` / ``close``. A checkpoint that silently failed is the worst
  failure mode for a fault-tolerant trainer: you only find out when you need it.
* **Drain on close.** ``close`` finishes every queued write; nothing is dropped.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, Optional

from .snapshot import SnapshotBuffers
from .storage import StorageBackend

logger = logging.getLogger(__name__)


class CheckpointSaveError(RuntimeError):
    """A background checkpoint write failed."""


@dataclass
class SaveRecord:
    """Timing of one checkpoint save."""

    path: str
    nbytes: int
    snapshot_seconds: float  # time the training thread was blocked
    write_seconds: float  # background serialization + write (+ fsync)

    @property
    def write_throughput_mbps(self) -> float:
        return self.nbytes / 1e6 / self.write_seconds if self.write_seconds > 0 else float("nan")


def _save_with_retries(storage: StorageBackend, state: Dict[str, Any], path: str, retries: int) -> int:
    for attempt in range(retries + 1):
        try:
            return storage.save(state, path)
        except Exception as e:
            if attempt == retries:
                raise
            delay = 2 ** attempt
            logger.warning("Checkpoint write to %s failed (%s); retrying in %ss", path, e, delay)
            time.sleep(delay)
    raise AssertionError("unreachable")


class AsyncCheckpointWriter:
    """Snapshots state on the caller's thread and writes it on a background thread."""

    def __init__(
        self,
        storage: StorageBackend,
        max_pending: int = 1,
        pin_memory: bool = False,
        reuse_buffers: bool = True,
        write_retries: int = 0,
    ):
        if max_pending < 1:
            raise ValueError("max_pending must be >= 1")
        self.storage = storage
        self.max_pending = max_pending
        self.write_retries = write_retries
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="flextrain-ckpt-writer")
        self._slots = threading.Semaphore(max_pending)
        # One buffer set per in-flight slot. Because writes complete in FIFO order, once a
        # slot is acquired for save n, save n - max_pending (the last user of this buffer
        # set) has finished, so the buffers can be safely overwritten.
        self._ring = [SnapshotBuffers(pin_memory=pin_memory, reuse=reuse_buffers) for _ in range(max_pending)]
        self._submitted = 0
        self._inflight = 0
        self._cond = threading.Condition()
        self._failure: Optional[BaseException] = None
        self._closed = False
        self.history: Deque[SaveRecord] = deque(maxlen=100)

    # ------------------------------------------------------------------ public API
    def submit(
        self,
        state: Dict[str, Any],
        path: str,
        on_success: Optional[Callable[[str], None]] = None,
    ) -> Future:
        """Snapshot ``state`` (blocking) and schedule the write (non-blocking).

        ``on_success(path)`` runs on the writer thread after the write is durable.
        """
        self.raise_if_failed()
        if self._closed:
            raise RuntimeError("AsyncCheckpointWriter is closed")

        self._slots.acquire()  # backpressure
        counted = False
        try:
            buffers = self._ring[self._submitted % self.max_pending]
            start = time.perf_counter()
            snapshot = buffers.snapshot(state)
            snapshot_seconds = time.perf_counter() - start
            with self._cond:
                self._inflight += 1
                counted = True
            future = self._executor.submit(self._write, snapshot, path, on_success, snapshot_seconds)
        except BaseException:
            with self._cond:
                if counted:
                    self._inflight -= 1
                    self._cond.notify_all()
            self._slots.release()
            raise
        self._submitted += 1
        return future

    def wait(self) -> None:
        """Block until every submitted write has finished; re-raise any failure."""
        with self._cond:
            while self._inflight:
                self._cond.wait()
        self.raise_if_failed()

    def pending_count(self) -> int:
        with self._cond:
            return self._inflight

    def raise_if_failed(self) -> None:
        with self._cond:
            failure, self._failure = self._failure, None
        if failure is not None:
            raise CheckpointSaveError(f"background checkpoint write failed: {failure}") from failure

    def close(self, raise_errors: bool = True) -> None:
        """Drain all pending writes and stop the writer thread."""
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=True)
        for buffers in self._ring:
            buffers.release()
        if raise_errors:
            self.raise_if_failed()

    @property
    def host_buffer_bytes(self) -> int:
        return sum(b.nbytes for b in self._ring)

    # ------------------------------------------------------------------ worker side
    def _write(self, snapshot, path, on_success, snapshot_seconds) -> SaveRecord:
        try:
            start = time.perf_counter()
            nbytes = _save_with_retries(self.storage, snapshot, path, self.write_retries)
            record = SaveRecord(path, nbytes, snapshot_seconds, time.perf_counter() - start)
            self.history.append(record)
            logger.info("Checkpoint written: %s (%.1f MB in %.2fs, %.0f MB/s)",
                        path, nbytes / 1e6, record.write_seconds, record.write_throughput_mbps)
            if on_success is not None:
                try:
                    on_success(path)
                except Exception:  # retention problems must not fail a durable save
                    logger.exception("post-save hook failed for %s", path)
            return record
        except BaseException as e:
            logger.error("Async checkpoint write to %s failed: %s", path, e)
            with self._cond:
                if self._failure is None:
                    self._failure = e
            raise
        finally:
            del snapshot
            self._slots.release()
            with self._cond:
                self._inflight -= 1
                self._cond.notify_all()
