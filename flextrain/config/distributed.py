"""Distributed training configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .base import BaseConfig, _one_of, _require


@dataclass
class DistributedConfig(BaseConfig):
    """Process-group, model-wrapping and precision settings.

    Rank / world size are *not* configured here: they come from the launcher
    (``torchrun`` / ``flextrain launch``) through the standard environment variables.
    """

    strategy: str = "ddp"  # ddp | fsdp (ignored when world_size == 1)
    backend: str = "auto"  # auto | nccl | gloo  (auto: nccl on CUDA, else gloo)
    device: str = "auto"  # auto | cuda | cpu
    timeout_minutes: float = 30.0  # collective timeout: a hung peer raises instead of hanging forever

    # DDP
    find_unused_parameters: bool = False
    gradient_as_bucket_view: bool = True
    static_graph: bool = False

    # FSDP
    sharding_strategy: str = "FULL_SHARD"  # FULL_SHARD | SHARD_GRAD_OP | NO_SHARD | HYBRID_SHARD
    cpu_offload: bool = False
    backward_prefetch: str = "BACKWARD_PRE"  # BACKWARD_PRE | BACKWARD_POST | NONE
    forward_prefetch: bool = False
    limit_all_gathers: bool = True
    use_orig_params: bool = True
    # Module class names that become FSDP units and activation-checkpoint boundaries,
    # e.g. ["GPT2Block"]. If empty, FSDP falls back to a size-based wrap policy.
    wrap_modules: List[str] = field(default_factory=list)
    fsdp_min_num_params: int = 1_000_000

    # Mixed precision (CUDA only; bf16 needs no loss scaling, fp16 uses a GradScaler)
    mixed_precision: bool = True
    mixed_precision_dtype: str = "bf16"  # bf16 | fp16

    # Activation checkpointing of the ``wrap_modules`` blocks
    activation_checkpointing: bool = False

    @classmethod
    def section_name(cls) -> str:
        return "distributed"

    def validate(self) -> None:
        self.strategy = self.strategy.lower()
        self.backend = self.backend.lower()
        self.device = self.device.lower()
        self.mixed_precision_dtype = self.mixed_precision_dtype.lower()
        self.sharding_strategy = self.sharding_strategy.upper()
        self.backward_prefetch = self.backward_prefetch.upper()
        _one_of("distributed.strategy", self.strategy, {"ddp", "fsdp"})
        _one_of("distributed.backend", self.backend, {"auto", "nccl", "gloo"})
        _one_of("distributed.device", self.device, {"auto", "cuda", "cpu"})
        _one_of("distributed.mixed_precision_dtype", self.mixed_precision_dtype, {"bf16", "fp16"})
        _one_of("distributed.sharding_strategy", self.sharding_strategy,
                {"FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD", "HYBRID_SHARD"})
        _one_of("distributed.backward_prefetch", self.backward_prefetch, {"BACKWARD_PRE", "BACKWARD_POST", "NONE"})
        _require(self.timeout_minutes > 0, "distributed.timeout_minutes must be positive")
        _require(self.fsdp_min_num_params >= 1, "distributed.fsdp_min_num_params must be >= 1")
        _require(not (self.activation_checkpointing and not self.wrap_modules),
                 "distributed.activation_checkpointing requires distributed.wrap_modules "
                 "(the block classes to checkpoint)")
