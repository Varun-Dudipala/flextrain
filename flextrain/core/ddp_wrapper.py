"""DDP wrapper for distributed training."""

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from flextrain.config import DistributedConfig


class DDPWrapper:
    """Wrapper for PyTorch DistributedDataParallel."""

    @staticmethod
    def wrap(model: nn.Module, config: DistributedConfig) -> nn.Module:
        """Wrap model with DDP."""
        return DDP(
            model,
            find_unused_parameters=config.find_unused_parameters,
            gradient_as_bucket_view=config.gradient_as_bucket_view,
            static_graph=config.static_graph,
        )

    @staticmethod
    def unwrap(model: nn.Module) -> nn.Module:
        """Unwrap DDP model."""
        if isinstance(model, DDP):
            return model.module
        return model
