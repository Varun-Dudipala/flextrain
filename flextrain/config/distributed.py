"""Distributed training configuration."""

from dataclasses import dataclass
from typing import Optional, List
from .base import BaseConfig


@dataclass
class DistributedConfig(BaseConfig):
    """Configuration for distributed training."""

    strategy: str = "ddp"  # ddp | fsdp
    backend: str = "nccl"
    timeout_minutes: int = 30

    # DDP settings
    find_unused_parameters: bool = False
    gradient_as_bucket_view: bool = True
    static_graph: bool = False

    # FSDP settings
    sharding_strategy: str = "FULL_SHARD"
    cpu_offload: bool = False
    backward_prefetch: str = "BACKWARD_PRE"
    forward_prefetch: bool = False
    limit_all_gathers: bool = True
    use_orig_params: bool = True

    # Mixed precision
    mixed_precision: bool = True
    mixed_precision_dtype: str = "bf16"

    # Activation checkpointing
    activation_checkpointing: bool = False

    # Device
    device: str = "cuda"
    local_rank: int = -1

    def validate(self) -> None:
        valid_strategies = {"ddp", "fsdp"}
        if self.strategy.lower() not in valid_strategies:
            raise ValueError(f"strategy must be one of {valid_strategies}")
