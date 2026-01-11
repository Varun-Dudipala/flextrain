"""Training configuration."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from .base import BaseConfig


@dataclass
class TrainingConfig(BaseConfig):
    """Training hyperparameters configuration."""

    # Model
    model_name: str = "gpt2"
    model_config: Dict[str, Any] = field(default_factory=dict)

    # Batch size
    batch_size: int = 32
    micro_batch_size: Optional[int] = None

    # Optimizer
    optimizer: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    # LR Scheduler
    lr_scheduler: str = "cosine"
    warmup_steps: int = 100
    warmup_ratio: Optional[float] = None

    # Training duration
    max_steps: Optional[int] = None
    max_epochs: Optional[int] = None

    # Gradient
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1

    # Data
    dataset_path: Optional[str] = None
    num_workers: int = 4
    pin_memory: bool = True
    max_seq_length: int = 512

    # Logging
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 500

    def validate(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

    @property
    def effective_batch_size(self) -> int:
        micro = self.micro_batch_size or self.batch_size
        return micro * self.gradient_accumulation_steps
