"""Elastic launch and fault-tolerance configuration."""

from __future__ import annotations

import signal
from dataclasses import dataclass, field
from typing import List, Optional

from .base import BaseConfig, _require


@dataclass
class ElasticConfig(BaseConfig):
    """Settings translated into ``torchrun`` arguments by ``flextrain launch``.

    Elasticity follows the torchelastic model: when a worker fails or nodes join/leave,
    the elastic agent tears the worker group down and restarts it (up to ``max_restarts``)
    with the new world size. FlexTrain makes that restart cheap and correct: workers
    auto-resume from the newest valid checkpoint, re-shard the remaining data for the
    new world size, and re-derive gradient accumulation to keep the global batch fixed.
    """

    min_nodes: int = 1
    max_nodes: int = 1
    nproc_per_node: int = 1

    rdzv_backend: str = "c10d"
    rdzv_endpoint: Optional[str] = None  # host:port; required when max_nodes > 1
    rdzv_id: Optional[str] = None  # defaults to main.experiment_name

    max_restarts: int = 3
    monitor_interval: float = 5.0  # seconds between agent health checks of local workers
    # How long the agent waits, after a stop request reaches it, between telling the workers to
    # stop (SIGTERM) and killing them (SIGKILL). This is the window a preempted job has to finish
    # its step and write a blocking checkpoint, so it must cover that save; torchrun's own default
    # is 30. (After a *worker failure* torchrun always waits 30 s for the surviving workers; see
    # fault_tolerance.stop_grace_seconds for what shortens that.)
    shutdown_timeout: int = 60

    @classmethod
    def section_name(cls) -> str:
        return "elastic"

    def validate(self) -> None:
        _require(self.min_nodes >= 1, "elastic.min_nodes must be at least 1")
        _require(self.max_nodes >= self.min_nodes, "elastic.max_nodes must be >= elastic.min_nodes")
        _require(self.nproc_per_node >= 1, "elastic.nproc_per_node must be >= 1")
        _require(self.max_restarts >= 0, "elastic.max_restarts must be >= 0")
        _require(self.monitor_interval > 0, "elastic.monitor_interval must be positive")
        _require(self.shutdown_timeout >= 1, "elastic.shutdown_timeout must be >= 1")
        if self.rdzv_endpoint is not None:
            host, sep, port = self.rdzv_endpoint.rpartition(":")
            _require(bool(sep and host and port.isdigit()),
                     f"elastic.rdzv_endpoint must look like host:port, got {self.rdzv_endpoint!r}")



@dataclass
class FaultToleranceConfig(BaseConfig):
    """In-process failure handling."""

    # Preemption: these signals request a *coordinated* stop. Every rank finishes the
    # current optimizer step, a checkpoint is written, and train() returns "preempted".
    # A second SIGINT forces an immediate KeyboardInterrupt.
    handle_signals: bool = True
    signals: List[str] = field(default_factory=lambda: ["SIGTERM", "SIGINT", "SIGUSR1"])
    # How often (in optimizer steps) ranks agree on stop / time-based-save decisions.
    # Costs one tiny all-reduce per check.
    stop_check_interval: int = 1
    # After a stop signal, a rank that neither makes progress (finishes a micro-batch) nor
    # reaches the agreed stop within this many seconds is taken to be blocked on a dead or hung
    # peer and exits at once, instead of sitting out the agent's whole shutdown timeout. Raise it
    # if a single micro-batch can take longer; null disables the watchdog.
    stop_grace_seconds: Optional[float] = 15.0

    # Divergence guard: an optimizer step with a non-finite gradient norm is skipped.
    # More than ``max_consecutive_nonfinite`` skipped steps in a row raises, which lets
    # the elastic agent restart the job from the last good checkpoint.
    skip_nonfinite_steps: bool = True
    max_consecutive_nonfinite: int = 5

    @classmethod
    def section_name(cls) -> str:
        return "fault_tolerance"

    def validate(self) -> None:
        self.signals = [s.upper() for s in self.signals]
        for name in self.signals:
            _require(isinstance(getattr(signal, name, None), signal.Signals),
                     f"fault_tolerance.signals: unknown signal {name!r}")
            _require(name not in ("SIGKILL", "SIGSTOP"),
                     f"fault_tolerance.signals: {name} cannot be caught, so it cannot request a graceful stop")
        _require(self.stop_check_interval >= 1, "fault_tolerance.stop_check_interval must be >= 1")
        if self.stop_grace_seconds is not None:
            _require(self.stop_grace_seconds > 0, "fault_tolerance.stop_grace_seconds must be positive (or null)")
        _require(self.max_consecutive_nonfinite >= 1, "fault_tolerance.max_consecutive_nonfinite must be >= 1")
