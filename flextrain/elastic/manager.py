"""Elastic scaling manager."""

import logging
import time
from typing import Optional
import torch.distributed as dist

from flextrain.config import ElasticConfig

logger = logging.getLogger(__name__)


class ElasticManager:
    """Manages elastic scaling of training jobs."""

    def __init__(self, config: ElasticConfig):
        self.config = config
        self._last_scale_time = time.time()
        self._current_world_size = dist.get_world_size() if dist.is_initialized() else 1

    @property
    def world_size(self) -> int:
        return self._current_world_size

    def should_reconfigure(self) -> bool:
        """Check if reconfiguration is needed."""
        if not self.config.allow_elastic_scaling:
            return False

        if not dist.is_initialized():
            return False

        # Check cooldown
        if time.time() - self._last_scale_time < self.config.scale_cooldown_seconds:
            return False

        # Check if world size changed
        new_world_size = dist.get_world_size()
        if new_world_size != self._current_world_size:
            logger.info(f"World size changed: {self._current_world_size} -> {new_world_size}")
            return True

        return False

    def reconfigure(self) -> bool:
        """Reconfigure training for new world size."""
        if not dist.is_initialized():
            return False

        new_world_size = dist.get_world_size()
        old_world_size = self._current_world_size

        logger.info(f"Reconfiguring: {old_world_size} -> {new_world_size} workers")

        self._current_world_size = new_world_size
        self._last_scale_time = time.time()

        # Barrier to sync all workers
        dist.barrier()

        logger.info("Reconfiguration complete")
        return True

    def get_scale_factor(self) -> float:
        """Get the scaling factor for learning rate adjustment."""
        return self._current_world_size / self.config.min_nodes
