"""FSDP wrapping and activation checkpointing."""

from __future__ import annotations

import functools
import logging
from typing import Any, Optional, Set, Type

import torch
import torch.nn as nn
from torch.distributed.fsdp import (
    BackwardPrefetch,
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy

from flextrain.config import DistributedConfig

logger = logging.getLogger(__name__)

PRECISION_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


def resolve_module_classes(model: nn.Module, names) -> Set[Type[nn.Module]]:
    """Map class names from the config (e.g. ``"GPT2Block"``) to classes found in ``model``."""
    wanted = set(names)
    found = {type(m) for m in model.modules() if type(m).__name__ in wanted}
    missing = wanted - {cls.__name__ for cls in found}
    if missing:
        raise ValueError(f"distributed.wrap_modules: no module of class {sorted(missing)} in the model")
    return found


def apply_activation_checkpointing(model: nn.Module, module_names) -> int:
    """Recompute activations of the named block classes during backward. Returns #blocks."""
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        checkpoint_wrapper,
    )
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        apply_activation_checkpointing as _apply,
    )

    classes = tuple(resolve_module_classes(model, module_names))
    count = sum(1 for m in model.modules() if isinstance(m, classes))
    _apply(
        model,
        checkpoint_wrapper_fn=functools.partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
        check_fn=lambda m: isinstance(m, classes),
    )
    logger.info("Activation checkpointing enabled for %d block(s)", count)
    return count


class FSDPWrapper:
    """Wrapper for PyTorch FSDP (requires CUDA)."""

    SHARDING_STRATEGIES = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
        "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
    }
    BACKWARD_PREFETCH = {
        "BACKWARD_PRE": BackwardPrefetch.BACKWARD_PRE,
        "BACKWARD_POST": BackwardPrefetch.BACKWARD_POST,
        "NONE": None,
    }

    @staticmethod
    def auto_wrap_policy(model: nn.Module, config: DistributedConfig) -> Any:
        """Shard per transformer block if ``wrap_modules`` is given, else by parameter count."""
        if config.wrap_modules:
            classes = resolve_module_classes(model, config.wrap_modules)
            return functools.partial(transformer_auto_wrap_policy, transformer_layer_cls=classes)
        return functools.partial(size_based_auto_wrap_policy, min_num_params=config.fsdp_min_num_params)

    @staticmethod
    def mixed_precision(config: DistributedConfig) -> Optional[MixedPrecision]:
        if not config.mixed_precision:
            return None
        dtype = PRECISION_DTYPES[config.mixed_precision_dtype]
        # Gradients are reduced in fp32 for numerical stability.
        return MixedPrecision(param_dtype=dtype, reduce_dtype=torch.float32, buffer_dtype=dtype)

    @classmethod
    def wrap(
        cls,
        model: nn.Module,
        config: DistributedConfig,
        device: Optional[torch.device] = None,
        auto_wrap_policy: Optional[Any] = None,
    ) -> FSDP:
        if device is None or device.type != "cuda":
            raise RuntimeError("FSDP requires CUDA devices; use strategy 'ddp' for CPU training")
        return FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy or cls.auto_wrap_policy(model, config),
            sharding_strategy=cls.SHARDING_STRATEGIES[config.sharding_strategy],
            mixed_precision=cls.mixed_precision(config),
            cpu_offload=CPUOffload(offload_params=True) if config.cpu_offload else None,
            backward_prefetch=cls.BACKWARD_PREFETCH[config.backward_prefetch],
            forward_prefetch=config.forward_prefetch,
            limit_all_gathers=config.limit_all_gathers,
            use_orig_params=config.use_orig_params,
            device_id=device,
            sync_module_states=True,  # broadcast rank 0's initial weights
        )
