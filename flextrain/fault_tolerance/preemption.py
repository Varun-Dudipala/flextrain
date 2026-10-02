"""Graceful, coordinated handling of preemption signals.

The naive approach - save a checkpoint (or ``sys.exit``) inside the signal handler - is
wrong for distributed training:

* the handler runs between arbitrary bytecodes, possibly mid optimizer step, so the
  saved state may be inconsistent;
* usually only *some* ranks get the signal (one preempted node). If those ranks stop,
  the others block forever in the next collective.

Instead the handler only sets a flag. At step boundaries every rank calls
``should_stop()``, which all-reduces the flag (MAX), so all ranks agree to stop after
the *same* step, then they save one consistent checkpoint together and exit cleanly.
"""

from __future__ import annotations

import logging
import signal
import threading
from types import FrameType
from typing import Dict, Iterable, Optional

from flextrain.core.distributed import all_reduce_tensor

logger = logging.getLogger(__name__)


class PreemptionHandler:
    """Turns termination signals into a stop request polled at step boundaries."""

    def __init__(self, signals: Iterable[str] = ("SIGTERM", "SIGINT", "SIGUSR1")):
        self.signals = [getattr(signal, name) for name in signals if hasattr(signal, name)]
        self._received: Optional[int] = None
        self._previous: Dict[int, object] = {}
        self._installed = False

    @property
    def stop_requested(self) -> bool:
        """True if *this* process received a signal (not yet agreed across ranks)."""
        return self._received is not None

    @property
    def received_signal(self) -> Optional[str]:
        return signal.Signals(self._received).name if self._received is not None else None

    def request_stop(self, signum: int = signal.SIGTERM) -> None:
        """Programmatic stop request (same path as receiving a signal)."""
        self._received = signum

    def _handle(self, signum: int, frame: Optional[FrameType]) -> None:
        if signum == signal.SIGINT and self._received == signal.SIGINT:
            # Second Ctrl-C: give up on graceful shutdown.
            signal.signal(signal.SIGINT, signal.default_int_handler)
            raise KeyboardInterrupt
        self._received = signum
        # Logging from a signal handler is not strictly async-signal-safe, but CPython runs
        # handlers on the main thread between bytecodes, so this cannot deadlock on the GIL.
        logger.warning("Received %s: will checkpoint and stop at the next step boundary",
                       signal.Signals(signum).name)

    def install(self) -> bool:
        """Install handlers (main thread only). Returns False if not possible."""
        if self._installed:
            return True
        if threading.current_thread() is not threading.main_thread():
            logger.warning("Preemption handler not installed: not on the main thread")
            return False
        for sig in self.signals:
            self._previous[sig] = signal.signal(sig, self._handle)
        self._installed = True
        return True

    def uninstall(self) -> None:
        if not self._installed:
            return
        for sig, previous in self._previous.items():
            signal.signal(sig, previous)
        self._previous.clear()
        self._installed = False

    def should_stop(self) -> bool:
        """Collective: True on every rank iff any rank requested a stop."""
        return all_reduce_tensor([1.0 if self.stop_requested else 0.0], op="max")[0] > 0

    def __enter__(self) -> PreemptionHandler:
        self.install()
        return self

    def __exit__(self, *exc) -> None:
        self.uninstall()
