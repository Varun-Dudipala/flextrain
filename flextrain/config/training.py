"""Training configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .base import BaseConfig, _one_of, _require


@dataclass
class TrainingConfig(BaseConfig):
    """Optimization hyperparameters and loop control.

    Batch semantics:
      * ``batch_size`` is the per-device micro-batch fed to one forward pass.
      * One optimizer step consumes ``batch_size * gradient_accumulation_steps * world_size``
        samples (the *global* batch).
      * If ``global_batch_size`` is set, ``gradient_accumulation_steps`` is derived at runtime
        from the world size, so the optimization is unchanged when an elastic job is resized.
    """

    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    global_batch_size: Optional[int] = None

    # Optimizer
    optimizer: str = "adamw"  # adamw | adam | sgd
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    momentum: float = 0.9  # sgd only

    # LR schedule: linear warmup, then decay to ``learning_rate * min_lr_ratio``
    lr_scheduler: str = "cosine"  # cosine | linear | constant
    warmup_steps: int = 0
    min_lr_ratio: float = 0.0

    # Duration: at least one of these must be set before training
    max_steps: Optional[int] = None
    max_epochs: Optional[int] = None

    # Gradients (0 disables clipping; the norm is still computed and logged)
    max_grad_norm: float = 1.0

    # Data loading
    num_workers: int = 2
    pin_memory: bool = True

    # Logging / evaluation (eval_interval == 0 disables periodic evaluation)
    log_interval: int = 10
    eval_interval: int = 0
    eval_max_batches: Optional[int] = None

    @classmethod
    def section_name(cls) -> str:
        return "training"

    def validate(self) -> None:
        self.optimizer = self.optimizer.lower()
        self.lr_scheduler = self.lr_scheduler.lower()
        _require(self.batch_size >= 1, "training.batch_size must be >= 1")
        _require(self.gradient_accumulation_steps >= 1, "training.gradient_accumulation_steps must be >= 1")
        if self.global_batch_size is not None:
            _require(self.global_batch_size >= self.batch_size,
                     "training.global_batch_size must be >= training.batch_size")
        _one_of("training.optimizer", self.optimizer, {"adamw", "adam", "sgd"})
        _one_of("training.lr_scheduler", self.lr_scheduler, {"cosine", "linear", "constant"})
        _require(self.learning_rate > 0, "training.learning_rate must be positive")
        _require(self.weight_decay >= 0, "training.weight_decay must be >= 0")
        _require(0 <= self.beta1 < 1 and 0 <= self.beta2 < 1, "training.beta1/beta2 must be in [0, 1)")
        _require(self.eps > 0, "training.eps must be positive")
        _require(self.warmup_steps >= 0, "training.warmup_steps must be >= 0")
        _require(0.0 <= self.min_lr_ratio <= 1.0, "training.min_lr_ratio must be in [0, 1]")
        if self.max_steps is not None:
            _require(self.max_steps >= 1, "training.max_steps must be >= 1")
            _require(self.warmup_steps <= self.max_steps, "training.warmup_steps must be <= training.max_steps")
        if self.max_epochs is not None:
            _require(self.max_epochs >= 1, "training.max_epochs must be >= 1")
        _require(self.max_grad_norm >= 0, "training.max_grad_norm must be >= 0")
        _require(self.num_workers >= 0, "training.num_workers must be >= 0")
        _require(self.log_interval >= 1, "training.log_interval must be >= 1")
        _require(self.eval_interval >= 0, "training.eval_interval must be >= 0")
        if self.eval_max_batches is not None:
            _require(self.eval_max_batches >= 1, "training.eval_max_batches must be >= 1")

    def accumulation_steps_for(self, world_size: int) -> int:
        """Gradient accumulation steps to use at ``world_size``.

        With ``global_batch_size`` set this is re-derived so the global batch stays constant;
        if it does not divide evenly the nearest value is used (see ``global_batch_for``).
        """
        if self.global_batch_size is None:
            return self.gradient_accumulation_steps
        per_step = self.batch_size * world_size
        return max(1, round(self.global_batch_size / per_step))

    def global_batch_for(self, world_size: int) -> int:
        """Samples consumed per optimizer step at ``world_size``."""
        return self.batch_size * self.accumulation_steps_for(world_size) * world_size
