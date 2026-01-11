"""Fault tolerance handler."""

import logging
import time
from typing import Any, Optional
import torch
import torch.nn as nn
import torch.distributed as dist

from flextrain.config import ElasticConfig

logger = logging.getLogger(__name__)


class FaultHandler:
    """Handles fault detection and recovery."""

    def __init__(self, config: ElasticConfig, checkpoint_manager: Any = None):
        self.config = config
        self.checkpoint_manager = checkpoint_manager
        self._last_heartbeat = time.time()
        self._failure_count = 0

    def health_check(self) -> bool:
        """Perform health check on all workers."""
        if not dist.is_initialized():
            return True

        try:
            # Simple all-gather to check connectivity
            tensor = torch.tensor([1.0], device="cuda" if torch.cuda.is_available() else "cpu")
            gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, tensor, async_op=False)
            self._last_heartbeat = time.time()
            return True
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False

    def handle_failure(self, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None) -> bool:
        """Handle a detected failure."""
        self._failure_count += 1
        logger.warning(f"Handling failure #{self._failure_count}")

        if self._failure_count > self.config.max_restarts:
            logger.error("Max restarts exceeded")
            return False

        # Try to recover from checkpoint
        if self.checkpoint_manager:
            try:
                result = self.checkpoint_manager.try_load_latest(model, optimizer)
                if result:
                    logger.info(f"Recovered from checkpoint at step {result.get('step', 0)}")
                    return True
            except Exception as e:
                logger.error(f"Recovery failed: {e}")

        return False

    def register_signal_handlers(self) -> None:
        """Register signal handlers for graceful shutdown."""
        import signal

        def handler(signum, frame):
            logger.warning(f"Received signal {signum}, initiating graceful shutdown")
            self._save_emergency_checkpoint()
            raise SystemExit(0)

        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)

    def _save_emergency_checkpoint(self) -> None:
        """Save emergency checkpoint on shutdown signal."""
        if self.checkpoint_manager is None:
            logger.warning("No checkpoint manager available for emergency save")
            return

        try:
            # Wait for any pending async saves first
            self.checkpoint_manager.wait_for_pending()
            logger.info("Emergency checkpoint: pending saves completed")
        except Exception as e:
            logger.error(f"Failed to complete pending saves: {e}")
