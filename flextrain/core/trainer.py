"""Main distributed trainer implementation."""

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from flextrain.config import Config
from flextrain.checkpoint import CheckpointManager
from flextrain.fault_tolerance import FaultHandler
from flextrain.tracking import MetricsLogger

logger = logging.getLogger(__name__)


@dataclass
class StepMetrics:
    """Metrics from a training step."""
    loss: float
    lr: float
    grad_norm: Optional[float] = None
    throughput: Optional[float] = None
    step_time: Optional[float] = None


class DistributedTrainer:
    """
    Main trainer class that orchestrates distributed training.

    Features:
    - DDP/FSDP model wrapping
    - Gradient accumulation
    - Mixed precision training
    - Async checkpointing
    - Fault tolerance
    - Metrics logging
    """

    def __init__(
        self,
        model: nn.Module,
        config: Config,
        train_dataset: Dataset,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
        eval_dataset: Optional[Dataset] = None,
    ):
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        # Distributed setup
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.is_main_process = self.rank == 0
        self.device = torch.device(f"cuda:{self.rank}" if torch.cuda.is_available() else "cpu")

        # Move model to device
        self.model = self.model.to(self.device)

        # Wrap model for distributed training
        self._wrap_model()

        # Create optimizer if not provided
        self.optimizer = optimizer or self._create_optimizer()
        self.lr_scheduler = lr_scheduler or self._create_scheduler()

        # Create data loaders
        self.train_loader = self._create_dataloader(train_dataset, shuffle=True)
        self.eval_loader = self._create_dataloader(eval_dataset, shuffle=False) if eval_dataset else None

        # Create checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            config.checkpoint,
            rank=self.rank,
            world_size=self.world_size,
        )

        # Create fault handler
        self.fault_handler = FaultHandler(
            config.elastic,
            checkpoint_manager=self.checkpoint_manager,
        )

        # Create metrics logger
        self.metrics_logger = MetricsLogger(
            experiment_name=config.main.experiment_name,
            run_name=config.main.run_name,
        )

        # Training state
        self.global_step = 0
        self.epoch = 0
        self._grad_scaler = None

        if config.distributed.mixed_precision:
            self._grad_scaler = torch.amp.GradScaler("cuda")

        logger.info(f"Trainer initialized on rank {self.rank}/{self.world_size}")

    def _wrap_model(self) -> None:
        """Wrap model for distributed training."""
        if self.world_size <= 1:
            return

        strategy = self.config.distributed.strategy.lower()

        if strategy == "ddp":
            from .ddp_wrapper import DDPWrapper
            self.model = DDPWrapper.wrap(self.model, self.config.distributed)
        elif strategy == "fsdp":
            from .fsdp_wrapper import FSDPWrapper
            self.model = FSDPWrapper.wrap(self.model, self.config.distributed)

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer."""
        tc = self.config.training
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=tc.learning_rate,
            weight_decay=tc.weight_decay,
            betas=(tc.beta1, tc.beta2),
            eps=tc.eps,
        )

    def _create_scheduler(self) -> Optional[Any]:
        """Create learning rate scheduler."""
        tc = self.config.training
        if tc.max_steps:
            total_steps = tc.max_steps
        else:
            return None

        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - tc.warmup_steps,
        )

    def _create_dataloader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        """Create distributed data loader."""
        sampler = None
        if self.world_size > 1:
            sampler = DistributedSampler(dataset, shuffle=shuffle)
            shuffle = False

        return DataLoader(
            dataset,
            batch_size=self.config.training.micro_batch_size or self.config.training.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.config.training.num_workers,
            pin_memory=self.config.training.pin_memory,
            drop_last=True,
        )

    def train_step(self, batch: Dict[str, torch.Tensor]) -> StepMetrics:
        """Execute single training step."""
        self.model.train()
        start_time = time.time()

        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # Forward pass with optional mixed precision
        with torch.amp.autocast("cuda", enabled=self.config.distributed.mixed_precision):
            outputs = self.model(**batch)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[1]
            loss = loss / self.config.training.gradient_accumulation_steps

        # Backward pass
        if self._grad_scaler:
            self._grad_scaler.scale(loss).backward()
        else:
            loss.backward()

        # Get current learning rate
        lr = self.optimizer.param_groups[0]["lr"]

        step_time = time.time() - start_time

        return StepMetrics(
            loss=loss.item() * self.config.training.gradient_accumulation_steps,
            lr=lr,
            step_time=step_time,
        )

    def optimizer_step(self) -> Optional[float]:
        """Execute optimizer step with gradient clipping."""
        grad_norm = None

        if self.config.training.max_grad_norm:
            if self._grad_scaler:
                self._grad_scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.max_grad_norm,
            ).item()

        if self._grad_scaler:
            self._grad_scaler.step(self.optimizer)
            self._grad_scaler.update()
        else:
            self.optimizer.step()

        self.optimizer.zero_grad()

        if self.lr_scheduler:
            self.lr_scheduler.step()

        return grad_norm

    def train(self) -> Dict[str, Any]:
        """Main training loop."""
        logger.info("Starting training...")

        # Try to resume from checkpoint
        resumed = self.checkpoint_manager.try_load_latest(
            self.model, self.optimizer, self.lr_scheduler
        )
        if resumed:
            self.global_step = resumed.get("step", 0)
            self.epoch = resumed.get("epoch", 0)
            logger.info(f"Resumed from step {self.global_step}")

        tc = self.config.training
        accumulation_steps = tc.gradient_accumulation_steps

        while True:
            self.epoch += 1

            if hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(self.epoch)

            for batch_idx, batch in enumerate(self.train_loader):
                # Health check
                if not self.fault_handler.health_check():
                    self._handle_failure()
                    continue

                # Training step
                metrics = self.train_step(batch)

                # Optimizer step after accumulation
                if (batch_idx + 1) % accumulation_steps == 0:
                    grad_norm = self.optimizer_step()
                    self.global_step += 1

                    # Logging
                    if self.global_step % tc.log_interval == 0:
                        self._log_metrics(metrics, grad_norm)

                    # Checkpointing
                    if self.global_step % tc.save_interval == 0:
                        self._save_checkpoint(metrics)

                    # Check termination
                    if tc.max_steps and self.global_step >= tc.max_steps:
                        return self._finalize_training()

            if tc.max_epochs and self.epoch >= tc.max_epochs:
                return self._finalize_training()

    def _log_metrics(self, metrics: StepMetrics, grad_norm: Optional[float]) -> None:
        """Log training metrics."""
        if self.is_main_process:
            self.metrics_logger.log({
                "train/loss": metrics.loss,
                "train/lr": metrics.lr,
                "train/grad_norm": grad_norm,
                "train/step_time": metrics.step_time,
            }, step=self.global_step)

            logger.info(
                f"Step {self.global_step}: loss={metrics.loss:.4f}, "
                f"lr={metrics.lr:.2e}, grad_norm={grad_norm:.2f}"
            )

    def _save_checkpoint(self, metrics: StepMetrics) -> None:
        """Save checkpoint."""
        self.checkpoint_manager.save(
            self.model,
            self.optimizer,
            self.lr_scheduler,
            step=self.global_step,
            epoch=self.epoch,
            metrics={"loss": metrics.loss},
        )

    def _handle_failure(self) -> None:
        """Handle training failure."""
        logger.warning("Handling failure, attempting recovery...")
        self.fault_handler.handle_failure(self.model, self.optimizer)

    def _finalize_training(self) -> Dict[str, Any]:
        """Finalize training and return results."""
        logger.info(f"Training completed at step {self.global_step}")

        # Final checkpoint
        self._save_checkpoint(StepMetrics(loss=0, lr=0))

        return {
            "global_step": self.global_step,
            "epoch": self.epoch,
        }
