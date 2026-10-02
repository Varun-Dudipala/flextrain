"""Runtime side of elastic training.

Under torchelastic a membership change (failure, scale up/down) restarts the whole
worker group with a possibly different world size. Each new incarnation must:

1. know it is a restart (``TORCHELASTIC_RESTART_COUNT``) and resume from a checkpoint,
2. re-derive gradient accumulation so the *global* batch - and therefore the
   optimization trajectory and LR schedule - is unchanged by the new world size,
3. continue the data stream from the same global position (see
   ``ResumableDistributedSampler``).

``ElasticManager`` owns (1) and (2) and reports what changed on resume.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from flextrain.config import TrainingConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ElasticEnv:
    """What the elastic agent tells this worker through the environment."""

    restart_count: int = 0
    max_restarts: Optional[int] = None
    run_id: Optional[str] = None
    local_world_size: Optional[int] = None
    group_rank: Optional[int] = None  # node index in this incarnation

    @classmethod
    def from_env(cls) -> ElasticEnv:
        def _int(name: str) -> Optional[int]:
            value = os.environ.get(name)
            return int(value) if value not in (None, "") else None

        return cls(
            restart_count=_int("TORCHELASTIC_RESTART_COUNT") or 0,
            max_restarts=_int("TORCHELASTIC_MAX_RESTARTS"),
            run_id=os.environ.get("TORCHELASTIC_RUN_ID"),
            local_world_size=_int("LOCAL_WORLD_SIZE"),
            group_rank=_int("GROUP_RANK"),
        )

    @property
    def under_elastic_agent(self) -> bool:
        return self.run_id is not None or self.max_restarts is not None

    @property
    def is_restart(self) -> bool:
        return self.restart_count > 0


@dataclass(frozen=True)
class TopologyChange:
    previous_world_size: int
    world_size: int
    previous_global_batch: Optional[int]
    global_batch: int

    @property
    def resized(self) -> bool:
        return self.previous_world_size != self.world_size

    @property
    def global_batch_changed(self) -> bool:
        return self.previous_global_batch is not None and self.previous_global_batch != self.global_batch


class ElasticManager:
    """Derives per-incarnation batch geometry and validates resumes across resizes."""

    def __init__(self, training: TrainingConfig, world_size: int, env: Optional[ElasticEnv] = None):
        self.training = training
        self.world_size = world_size
        self.env = env or ElasticEnv.from_env()

        self.accumulation_steps = training.accumulation_steps_for(world_size)
        self.global_batch = training.global_batch_for(world_size)
        self.samples_per_rank_per_step = training.batch_size * self.accumulation_steps
        if training.global_batch_size is not None and self.global_batch != training.global_batch_size:
            logger.warning(
                "global_batch_size=%d is not divisible by batch_size(%d) x world_size(%d); "
                "using gradient_accumulation_steps=%d -> effective global batch %d",
                training.global_batch_size, training.batch_size, world_size,
                self.accumulation_steps, self.global_batch,
            )

    def describe(self) -> Dict[str, Any]:
        return {
            "world_size": self.world_size,
            "gradient_accumulation_steps": self.accumulation_steps,
            "global_batch": self.global_batch,
            "restart_count": self.env.restart_count,
        }

    def on_resume(self, checkpoint_meta: Dict[str, Any]) -> TopologyChange:
        """Log and return how this incarnation differs from the one that checkpointed."""
        change = TopologyChange(
            previous_world_size=int(checkpoint_meta.get("world_size", self.world_size)),
            world_size=self.world_size,
            previous_global_batch=checkpoint_meta.get("global_batch"),
            global_batch=self.global_batch,
        )
        if change.resized:
            logger.warning(
                "Elastic resize: world size %d -> %d; gradient accumulation now %d (global batch %d)",
                change.previous_world_size, change.world_size, self.accumulation_steps, self.global_batch,
            )
        if change.global_batch_changed:
            logger.warning(
                "Global batch changed on resume (%d -> %d): the optimization differs from the "
                "original run. Set training.global_batch_size to keep it fixed across resizes.",
                change.previous_global_batch, change.global_batch,
            )
        return change
