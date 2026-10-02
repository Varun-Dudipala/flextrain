"""Fault tolerance: coordinated preemption handling and divergence guards.

Process-level failures (a crashed or hung rank, a lost node) are handled by the elastic
agent (``flextrain launch`` -> ``torchrun``): the collective timeout turns a hang into an
error, the agent restarts the worker group, and the trainer resumes from the newest
valid checkpoint. See ``flextrain.elastic``.
"""

from .guards import NonFiniteGuard, TrainingDivergedError
from .preemption import PreemptionHandler

__all__ = ["NonFiniteGuard", "PreemptionHandler", "TrainingDivergedError"]
