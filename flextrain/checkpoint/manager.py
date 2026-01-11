"""Checkpoint Manager - Main orchestrator for checkpoint operations."""

from typing import Any, Dict, Optional
from pathlib import Path
import time
import logging
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from flextrain.config import CheckpointConfig
from .storage import StorageBackend, create_storage_backend

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages checkpoint saving, loading, versioning, and pruning."""

    def __init__(self, config: CheckpointConfig, rank: int = 0, world_size: int = 1):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.is_main_process = rank == 0

        self.storage = create_storage_backend(
            backend_type=config.storage_backend,
            bucket_name=config.bucket_name,
        )

        self._checkpoints = []
        self._save_count = 0
        self._last_save_time = time.time()

        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"CheckpointManager initialized (backend={config.storage_backend})")

    def save(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
        step: int = 0,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        is_fsdp: bool = False,
        blocking: bool = True,
    ) -> Optional[str]:
        """Save checkpoint."""
        if not self.is_main_process and not is_fsdp:
            return None

        metrics = metrics or {}
        start_time = time.time()

        # Generate path
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"checkpoint_step{step:08d}_{timestamp}.pt"
        ckpt_path = str(Path(self.config.checkpoint_dir) / filename)

        # Get model state
        if isinstance(model, FSDP):
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType
            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
                model_state = model.state_dict()
        else:
            model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()

        state = {
            "model": model_state,
            "optimizer": optimizer.state_dict() if optimizer else None,
            "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler else None,
            "step": step,
            "epoch": epoch,
            "metrics": metrics,
            "config": self.config.to_dict() if hasattr(self.config, 'to_dict') else {},
        }

        self.storage.save(state, ckpt_path)
        self._checkpoints.append({"path": ckpt_path, "step": step, "metrics": metrics})
        self._prune_checkpoints()

        save_time = time.time() - start_time
        logger.info(f"Checkpoint saved: {ckpt_path} (step={step}, time={save_time*1000:.1f}ms)")

        return ckpt_path

    def load(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
        checkpoint_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load checkpoint."""
        if checkpoint_path is None:
            checkpoint_path = self._get_latest()

        if checkpoint_path is None:
            raise ValueError("No checkpoint found to load")

        logger.info(f"Loading checkpoint: {checkpoint_path}")
        state = self.storage.load(checkpoint_path)

        # Load model
        if isinstance(model, FSDP):
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType
            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
                model.load_state_dict(state["model"])
        else:
            target = model.module if hasattr(model, 'module') else model
            target.load_state_dict(state["model"])

        # Load optimizer
        if optimizer and state.get("optimizer"):
            optimizer.load_state_dict(state["optimizer"])

        # Load scheduler
        if lr_scheduler and state.get("lr_scheduler"):
            lr_scheduler.load_state_dict(state["lr_scheduler"])

        return {
            "step": state.get("step", 0),
            "epoch": state.get("epoch", 0),
            "metrics": state.get("metrics", {}),
        }

    def try_load_latest(
        self,
        model: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        """Try to load latest checkpoint if exists."""
        latest = self._get_latest()
        if latest is None:
            return None
        if model is not None:
            return self.load(model, optimizer, lr_scheduler, latest)
        return {"checkpoint_path": latest}

    def _get_latest(self) -> Optional[str]:
        """Get path to most recent checkpoint."""
        ckpt_dir = Path(self.config.checkpoint_dir)
        if not ckpt_dir.exists():
            return None

        checkpoints = list(ckpt_dir.glob("checkpoint_*.pt"))
        if not checkpoints:
            return None

        return str(max(checkpoints, key=lambda p: p.stat().st_mtime))

    def _prune_checkpoints(self) -> None:
        """Remove old checkpoints beyond retention limit."""
        if len(self._checkpoints) <= self.config.keep_last_n:
            return

        sorted_ckpts = sorted(self._checkpoints, key=lambda x: x["step"], reverse=True)
        to_keep = sorted_ckpts[:self.config.keep_last_n]
        to_delete = sorted_ckpts[self.config.keep_last_n:]

        for ckpt in to_delete:
            try:
                self.storage.delete(ckpt["path"])
                self._checkpoints.remove(ckpt)
            except Exception as e:
                logger.warning(f"Failed to delete checkpoint: {e}")

    def wait_for_pending(self) -> None:
        """Wait for any pending async saves."""
        pass
