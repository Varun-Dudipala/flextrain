"""Guards against numerically diverged training steps."""

from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)


class TrainingDivergedError(RuntimeError):
    """Raised after too many consecutive non-finite optimizer steps."""


class NonFiniteGuard:
    """Skips optimizer steps whose gradient norm is NaN/Inf; raises if it persists.

    The decision uses the *global* gradient norm, which is identical on every rank
    after gradient all-reduce (DDP) or the sharded norm reduction (FSDP), so all ranks
    skip the same steps without an extra collective.

    Raising after ``max_consecutive`` skips lets the elastic agent restart the job from
    the last good checkpoint instead of training on garbage for hours.
    """

    def __init__(self, max_consecutive: int = 5, enabled: bool = True):
        self.max_consecutive = max_consecutive
        self.enabled = enabled
        self.consecutive = 0
        self.total_skipped = 0

    def should_skip(self, grad_norm: float, step: int) -> bool:
        if not self.enabled or math.isfinite(grad_norm):
            self.consecutive = 0
            return False
        self.consecutive += 1
        self.total_skipped += 1
        logger.warning("Non-finite gradient norm at step %d; skipping update (%d consecutive)",
                       step, self.consecutive)
        if self.consecutive > self.max_consecutive:
            raise TrainingDivergedError(
                f"{self.consecutive} consecutive non-finite steps (limit {self.max_consecutive}) at step {step}"
            )
        return True

    def state_dict(self) -> dict:
        return {"consecutive": self.consecutive, "total_skipped": self.total_skipped}

    def load_state_dict(self, state: dict) -> None:
        self.consecutive = int(state.get("consecutive", 0))
        self.total_skipped = int(state.get("total_skipped", 0))
