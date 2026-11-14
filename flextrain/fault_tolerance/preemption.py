"""Graceful, coordinated handling of preemption signals.

The naive approach - save a checkpoint (or ``sys.exit``) inside the signal handler - is
wrong for distributed training:

* the handler runs between arbitrary bytecodes, possibly mid optimizer step, so the
  saved state may be inconsistent;
* usually only *some* ranks get the signal (one preempted node). If those ranks stop,
  the others block forever in the next collective.

Instead the handler only sets a flag. At step boundaries every rank all-reduces the flag
(MAX), so all ranks agree to stop after the *same* step, then they save one consistent
checkpoint together and exit cleanly.

Taking over a signal has a cost that must be paid back. A Python handler only runs when the
main thread returns to the interpreter. A rank that is blocked inside a C++ collective - its
peer died, which is exactly when the elastic agent sends SIGTERM to tear the group down -
never gets there, so with a handler installed it ignores the signal it would otherwise have
died from, and the agent has to wait out its whole shutdown timeout before it can SIGKILL the
rank and restart the job. The handler therefore comes with a watchdog thread: once a signal has
arrived, a rank that neither makes progress nor reaches the agreed stop within
``grace_seconds`` exits on the spot, as it would have without the handler.

The watchdog cannot learn about the signal from the Python handler - that is the very thing
that does not run while the main thread is stuck. It reads a wakeup pipe instead
(``signal.set_wakeup_fd``): CPython's C-level handler writes the signal number there the moment
the signal is delivered, whatever the main thread is doing.
"""

from __future__ import annotations

import contextlib
import logging
import os
import signal
import threading
import time
from types import FrameType
from typing import Dict, Iterable, Iterator, Optional

from flextrain.core.distributed import all_reduce_tensor

logger = logging.getLogger(__name__)

_WATCHDOG_POLL_SECONDS = 0.2
_active: Optional[PreemptionHandler] = None  # the handler whose wakeup pipe is installed, if any


def _reset_in_forked_child() -> None:
    """A forked child (a DataLoader worker) inherits the handlers and the wakeup pipe.

    Left alone, a signal sent to the child would be written into the *parent's* pipe and read by
    the parent's watchdog as its own, and the child would ignore SIGTERM. Give the child back the
    default behaviour; it is not a rank and has nothing to coordinate.
    """
    handler = _active
    if handler is None:
        return
    try:
        signal.set_wakeup_fd(-1)
        for sig in handler.signals:
            signal.signal(sig, signal.SIG_DFL)
    except ValueError:  # forked from a thread other than the main one: nothing we may change
        pass


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_in_forked_child)


class PreemptionHandler:
    """Turns termination signals into a stop request polled at step boundaries.

    Args:
        signals: names of the signals to take over.
        grace_seconds: how long after a signal this process may go without ``heartbeat()``
            before the watchdog concludes it is stuck and exits it. ``None`` disables the
            watchdog (the handler then behaves like a plain flag).
    """

    def __init__(self, signals: Iterable[str] = ("SIGTERM", "SIGINT", "SIGUSR1"),
                 grace_seconds: Optional[float] = None):
        self.signals = [getattr(signal, name) for name in signals if hasattr(signal, name)]
        self.grace_seconds = grace_seconds
        self._received: Optional[int] = None  # set by the Python handler (needs the main thread)
        self._received_at = 0.0
        self._delivered: Optional[int] = None  # set by the watchdog from the wakeup pipe (does not)
        self._delivered_at = 0.0
        self._wakeup_fds: Optional[tuple] = None
        self._previous_wakeup_fd = -1
        self._last_progress = time.monotonic()
        self._busy = 0
        self._agreed = False
        self._previous: Dict[int, object] = {}
        self._installed = False
        self._watchdog: Optional[threading.Thread] = None
        self._watching = False

    @property
    def stop_requested(self) -> bool:
        """True if *this* process received a signal (not yet agreed across ranks)."""
        return self._received is not None

    @property
    def received_signal(self) -> Optional[str]:
        return signal.Signals(self._received).name if self._received is not None else None

    def request_stop(self, signum: int = signal.SIGTERM) -> None:
        """Programmatic stop request (same path as receiving a signal)."""
        self._received_at = self._delivered_at = time.monotonic()
        self._received = self._delivered = signum

    # ------------------------------------------------------------------ liveness, for the watchdog
    def heartbeat(self) -> None:
        """Call whenever the training loop makes progress (each micro-batch, each eval batch)."""
        self._last_progress = time.monotonic()

    def stop_agreed(self) -> None:
        """Call once all ranks have agreed to stop: from here on this process is saving, not stuck."""
        self._agreed = True

    @contextlib.contextmanager
    def busy(self) -> Iterator[None]:
        """Mark a section that legitimately makes no progress for a long time (a checkpoint save)."""
        self._busy += 1
        try:
            yield
        finally:
            self._busy -= 1
            self.heartbeat()

    def _handle(self, signum: int, frame: Optional[FrameType]) -> None:
        # Only plain attribute writes here: no locks, no logging (a handler can interrupt the
        # main thread while it holds the logging lock). The watchdog thread reports the signal.
        if signum == signal.SIGINT and self._received == signal.SIGINT:
            # Second Ctrl-C: give up on graceful shutdown.
            signal.signal(signal.SIGINT, signal.default_int_handler)
            raise KeyboardInterrupt
        self._received_at = time.monotonic()
        self._received = signum

    def _drain_wakeup_pipe(self) -> None:
        """Note a signal the C-level handler has reported, even if the Python handler has not run."""
        if self._wakeup_fds is None:
            return
        try:
            data = os.read(self._wakeup_fds[0], 64)
        except (BlockingIOError, OSError):
            return
        for signum in data:
            if signum in self.signals and self._delivered is None:
                self._delivered, self._delivered_at = signum, time.monotonic()

    def _watch(self) -> None:
        reported = False
        while self._watching:
            time.sleep(_WATCHDOG_POLL_SECONDS)
            self._drain_wakeup_pipe()
            signum = self._delivered
            if signum is None:
                continue
            if not reported:
                reported = True
                logger.warning("Received %s: will checkpoint and stop at the next step boundary",
                               signal.Signals(signum).name)
            if self.grace_seconds is None or self._agreed or self._busy:
                continue
            stalled = time.monotonic() - max(self._last_progress, self._delivered_at)
            if stalled > self.grace_seconds:
                logger.error("No progress for %.0fs after %s (blocked on a dead or hung peer?): exiting so the "
                             "job can be restarted", stalled, signal.Signals(signum).name)
                # os._exit: this thread cannot restore the default handler (only the main thread
                # may call signal.signal), and the main thread is the one that is stuck.
                os._exit(128 + signum)

    def install(self) -> bool:
        """Install handlers (main thread only). Returns False if not possible."""
        global _active
        if self._installed:
            return True
        if threading.current_thread() is not threading.main_thread():
            logger.warning("Preemption handler not installed: not on the main thread")
            return False
        for sig in self.signals:
            self._previous[sig] = signal.signal(sig, self._handle)
        read_fd, write_fd = os.pipe()
        os.set_blocking(read_fd, False)
        os.set_blocking(write_fd, False)
        self._wakeup_fds = (read_fd, write_fd)
        self._previous_wakeup_fd = signal.set_wakeup_fd(write_fd, warn_on_full_buffer=False)
        _active = self
        self._installed = True
        self._watching = True
        self._watchdog = threading.Thread(target=self._watch, daemon=True, name="flextrain-stop-watchdog")
        self._watchdog.start()
        return True

    def uninstall(self) -> None:
        global _active
        if not self._installed:
            return
        self._watching = False
        if self._watchdog is not None:
            self._watchdog.join(timeout=5 * _WATCHDOG_POLL_SECONDS)
        signal.set_wakeup_fd(self._previous_wakeup_fd)
        for sig, previous in self._previous.items():
            signal.signal(sig, previous)
        if self._wakeup_fds is not None:
            for fd in self._wakeup_fds:
                os.close(fd)
            self._wakeup_fds = None
        _active = None
        self._previous.clear()
        self._installed = False

    def should_stop(self) -> bool:
        """Collective: True on every rank iff any rank requested a stop."""
        stop = all_reduce_tensor([1.0 if self.stop_requested else 0.0], op="max")[0] > 0
        if stop:
            self.stop_agreed()
        return stop

    def __enter__(self) -> PreemptionHandler:
        self.install()
        return self

    def __exit__(self, *exc) -> None:
        self.uninstall()
