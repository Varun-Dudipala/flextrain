"""FSDP wrapper for distributed training."""

import functools
from typing import Any, Dict, Optional, Type

import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision, CPUOffload
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from flextrain.config import DistributedConfig


class FSDPWrapper:
    """Wrapper for PyTorch FSDP."""

    SHARDING_STRATEGIES = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
        "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
    }

    @classmethod
    def wrap(
        cls,
        model: nn.Module,
        config: DistributedConfig,
        auto_wrap_policy: Optional[Any] = None,
    ) -> FSDP:
        """Wrap model with FSDP."""

        # Get sharding strategy
        sharding_strategy = cls.SHARDING_STRATEGIES.get(
            config.sharding_strategy.upper(),
            ShardingStrategy.FULL_SHARD,
        )

        # Mixed precision config
        mixed_precision = None
        if config.mixed_precision:
            dtype = torch.bfloat16 if config.mixed_precision_dtype == "bf16" else torch.float16
            mixed_precision = MixedPrecision(
                param_dtype=dtype,
                reduce_dtype=dtype,
                buffer_dtype=dtype,
            )

        # CPU offload
        cpu_offload = CPUOffload(offload_params=True) if config.cpu_offload else None

        return FSDP(
            model,
            sharding_strategy=sharding_strategy,
            mixed_precision=mixed_precision,
            cpu_offload=cpu_offload,
            use_orig_params=config.use_orig_params,
            limit_all_gathers=config.limit_all_gathers,
            auto_wrap_policy=auto_wrap_policy,
        )

    @staticmethod
    def get_state_dict(model: FSDP, full_state: bool = True) -> Dict[str, Any]:
        """Get state dict from FSDP model."""
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        if full_state:
            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
                return model.state_dict()
        else:
            return model.state_dict()

    @staticmethod
    def load_state_dict(model: FSDP, state_dict: Dict[str, Any], full_state: bool = True) -> None:
        """Load state dict into FSDP model."""
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        if full_state:
            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
                model.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
