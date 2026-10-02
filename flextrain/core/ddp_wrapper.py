"""DDP wrapper for distributed training."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from flextrain.config import DistributedConfig


class DDPWrapper:
    """Wrapper for PyTorch DistributedDataParallel."""

    @staticmethod
    def wrap(model: nn.Module, config: DistributedConfig, device: Optional[torch.device] = None) -> DDP:
        """Wrap ``model`` (already on ``device``). DDP broadcasts rank 0's parameters."""
        device_ids = [device.index] if device is not None and device.type == "cuda" else None
        return DDP(
            model,
            device_ids=device_ids,
            find_unused_parameters=config.find_unused_parameters,
            gradient_as_bucket_view=config.gradient_as_bucket_view,
            static_graph=config.static_graph,
        )

    @staticmethod
    def unwrap(model: nn.Module) -> nn.Module:
        return model.module if isinstance(model, DDP) else model
