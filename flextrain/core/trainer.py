"""Fault-tolerant distributed trainer.

One ``Trainer`` runs unchanged on a single CPU, one GPU, or many nodes launched by
``torchrun`` / ``flextrain launch``. What it guarantees:

* **Exact resume.** A checkpoint holds model, optimizer, LR scheduler, grad scaler, the
  sampler's global data position and every rank's RNG state. Training N steps straight
  and training k steps, crashing, and resuming produce bit-identical weights (tested).
* **Elastic resume.** Resuming at a different world size continues from the same global
  sample, and with ``training.global_batch_size`` the global batch is unchanged.
* **Non-blocking checkpoints.** With ``checkpoint.async_save`` the step loop pauses only
  for a device->host snapshot; serialization and I/O run on a background thread.
* **Coordinated preemption.** SIGTERM/SIGUSR1/SIGINT on *any* rank makes *all* ranks stop
  after the same step and write one consistent checkpoint.
* **Divergence guard.** Non-finite gradient steps are skipped; persistent divergence
  raises so the elastic agent restarts from the last good checkpoint.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import re
import socket
import time
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler, SequentialSampler

from flextrain.checkpoint import CheckpointManager, capture_rng_state, restore_rng_state, restore_state
from flextrain.config import Config, ConfigError
from flextrain.elastic import ElasticManager
from flextrain.fault_tolerance import NonFiniteGuard, PreemptionHandler
from flextrain.tracking import COMPLETED, FAILED, PREEMPTED, RUNNING, MetricsLogger, RunRegistry, make_run_id
from flextrain.utils import set_seed, setup_logging, teardown_file_logging

from .data_loader import ResumableDistributedSampler, build_dataloader
from .distributed import (
    all_gather_object,
    all_reduce_tensor,
    broadcast_object,
    destroy_distributed,
    init_distributed,
    is_dist,
)
from .fsdp_wrapper import PRECISION_DTYPES, apply_activation_checkpointing

logger = logging.getLogger(__name__)

OptimizerArg = Union[torch.optim.Optimizer, Callable[..., torch.optim.Optimizer], None]
SchedulerArg = Union[Any, Callable[[torch.optim.Optimizer], Any], None]


@dataclass
class TrainResult:
    status: str  # "completed" | "preempted"
    global_step: int
    epoch: int
    run_id: str
    last_checkpoint: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)


def _move_to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, Mapping):
        return {k: _move_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(_move_to_device(v, device) for v in batch)
    return batch


_WRAPPER_COMPONENTS = {"_fsdp_wrapped_module", "_checkpoint_wrapped_module", "_orig_mod"}


def _canonical_name(name: str) -> str:
    """Parameter name with DDP / FSDP / activation-checkpoint / torch.compile wrapper components removed."""
    parts = name.split(".")
    if parts[0] == "module":  # DDP only ever adds a single leading "module."
        parts = parts[1:]
    return ".".join(p for p in parts if p not in _WRAPPER_COMPONENTS)


def _count_tokens(batch: Any) -> int:
    if isinstance(batch, Mapping) and isinstance(batch.get("input_ids"), torch.Tensor):
        return batch["input_ids"].numel()
    return 0


class Trainer:
    """Orchestrates distributed training with checkpointing and fault tolerance.

    The model's ``forward`` receives the batch (``model(**batch)`` for dicts,
    ``model(*batch)`` for tuples) and must return the loss as a scalar tensor, a dict /
    object with a ``loss`` entry, or a ``(logits, loss)`` tuple. Override
    ``compute_loss`` for anything else.

    ``optimizer`` / ``lr_scheduler`` may be instances or factories
    (``params -> Optimizer`` / ``optimizer -> scheduler``). Use factories with FSDP,
    because FSDP replaces the parameters the optimizer must reference.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Config,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        optimizer: OptimizerArg = None,
        lr_scheduler: SchedulerArg = None,
        collate_fn: Optional[Callable] = None,
        registry: Optional[RunRegistry] = None,
    ):
        config.validate()
        tc = config.training
        if tc.max_steps is None and tc.max_epochs is None:
            raise ConfigError("set training.max_steps or training.max_epochs (otherwise training never ends)")
        self.config = config

        # ---- distributed context
        self._owns_process_group = not (torch.distributed.is_available() and torch.distributed.is_initialized())
        self.dist = init_distributed(config.distributed)
        self.rank, self.world_size, self.device = self.dist.rank, self.dist.world_size, self.dist.device
        self.is_main_process = self.dist.is_main
        setup_logging(config.main.log_level, self.rank)
        set_seed(config.main.seed + self.rank)

        # ---- batch geometry (re-derived for this world size)
        self.elastic = ElasticManager(tc, self.world_size)
        self.accumulation_steps = self.elastic.accumulation_steps
        self.global_batch = self.elastic.global_batch

        # ---- precision
        self.amp_dtype = self._resolve_amp_dtype()

        # ---- model (weight-decay eligibility is decided on the unwrapped model: FSDP flattens
        # sharded parameters to 1-D views, so shapes after wrapping are meaningless)
        self._decay_param_names = {_canonical_name(n) for n, p in model.named_parameters() if p.ndim >= 2}
        self.model = self._prepare_model(model)
        self._autocast_enabled = self.amp_dtype is not None and not self._is_fsdp
        self.scaler = self._build_grad_scaler()

        # ---- data
        if len(train_dataset) < self.global_batch:
            raise ValueError(f"train_dataset has {len(train_dataset)} samples, fewer than one global batch "
                             f"({self.global_batch})")
        self.train_dataset = train_dataset
        self.sampler = ResumableDistributedSampler(
            len(train_dataset), num_replicas=self.world_size, rank=self.rank, shuffle=True,
            seed=config.main.seed, samples_per_step=tc.batch_size * self.accumulation_steps,
        )
        pin = tc.pin_memory and self.device.type == "cuda"
        self.train_loader = build_dataloader(train_dataset, self.sampler, tc.batch_size, tc.num_workers,
                                             pin, collate_fn, seed=config.main.seed + self.rank)
        self.eval_loader = self._build_eval_loader(eval_dataset, collate_fn, pin) if eval_dataset is not None else None

        # ---- optimization
        self.optimizer = self._build_optimizer(optimizer)
        self.total_steps = tc.max_steps or tc.max_epochs * self.sampler.steps_per_epoch
        self.lr_scheduler = self._build_scheduler(lr_scheduler)

        # ---- fault tolerance & checkpointing
        ft = config.fault_tolerance
        self.preemption = PreemptionHandler(ft.signals) if ft.handle_signals else None
        self.nonfinite_guard = NonFiniteGuard(ft.max_consecutive_nonfinite, enabled=ft.skip_nonfinite_steps)
        self.checkpoint_manager = CheckpointManager(
            config.checkpoint, rank=self.rank, world_size=self.world_size,
            checkpoint_dir=config.resolved_checkpoint_dir(), pin_memory=self.device.type == "cuda",
        )
        self.registry = (registry or RunRegistry()) if self.is_main_process else None

        # ---- state
        self.global_step = 0
        self.epoch = 0
        self.run_name: Optional[str] = config.main.run_name
        self.run_id: Optional[str] = None
        self.metrics_logger: Optional[MetricsLogger] = None
        self.resumed_from: Optional[str] = None
        self.last_metrics: Dict[str, float] = {}
        self._last_checkpoint: Optional[str] = None
        self._last_checkpoint_time = time.monotonic()
        self._reset_log_window()

        logger.info("Trainer ready: world_size=%d, device=%s, strategy=%s, global_batch=%d "
                    "(micro %d x accum %d x ranks %d), amp=%s",
                    self.world_size, self.device, config.distributed.strategy if is_dist() else "single",
                    self.global_batch, tc.batch_size, self.accumulation_steps, self.world_size,
                    self.amp_dtype)

    # ================================================================== setup helpers
    def _resolve_amp_dtype(self) -> Optional[torch.dtype]:
        dc = self.config.distributed
        if not dc.mixed_precision:
            return None
        if self.device.type != "cuda":
            logger.info("mixed_precision ignored: only supported on CUDA devices")
            return None
        dtype = PRECISION_DTYPES[dc.mixed_precision_dtype]
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            logger.warning("bf16 not supported on this GPU; falling back to fp16 with loss scaling")
            dtype = torch.float16
        return dtype

    @property
    def _is_fsdp(self) -> bool:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        return isinstance(self.model, FSDP)

    def _prepare_model(self, model: nn.Module) -> nn.Module:
        dc = self.config.distributed
        model = model.to(self.device)
        use_fsdp = is_dist() and dc.strategy == "fsdp"
        if dc.activation_checkpointing and not use_fsdp:
            apply_activation_checkpointing(model, dc.wrap_modules)
        if not is_dist():
            return model
        if use_fsdp:
            from .fsdp_wrapper import FSDPWrapper

            model = FSDPWrapper.wrap(model, dc, device=self.device)
            if dc.activation_checkpointing:
                apply_activation_checkpointing(model, dc.wrap_modules)
            return model
        from .ddp_wrapper import DDPWrapper

        return DDPWrapper.wrap(model, dc, device=self.device)

    def _build_grad_scaler(self):
        if self.amp_dtype != torch.float16:
            return None  # fp32 and bf16 need no loss scaling
        if self._is_fsdp:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            return ShardedGradScaler()
        return torch.amp.GradScaler("cuda")

    def _build_eval_loader(self, dataset: Dataset, collate_fn, pin: bool) -> DataLoader:
        sampler = (DistributedSampler(dataset, shuffle=False, drop_last=False) if is_dist()
                   else SequentialSampler(dataset))
        return DataLoader(dataset, batch_size=self.config.training.batch_size, sampler=sampler,
                          num_workers=self.config.training.num_workers, pin_memory=pin, collate_fn=collate_fn)

    def _build_optimizer(self, optimizer: OptimizerArg) -> torch.optim.Optimizer:
        if isinstance(optimizer, torch.optim.Optimizer):
            if self._is_fsdp:
                raise ValueError("with FSDP pass an optimizer factory (params -> Optimizer), not an instance")
            return optimizer
        named = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        if callable(optimizer):
            return optimizer([p for _, p in named])
        tc = self.config.training
        if self._is_fsdp and not self.config.distributed.use_orig_params:
            # Flat parameters mix weights and biases; decay cannot be split per tensor.
            groups = [{"params": [p for _, p in named], "weight_decay": tc.weight_decay}]
        else:
            # No weight decay on biases and normalization weights (standard practice).
            decay = [p for n, p in named if _canonical_name(n) in self._decay_param_names]
            no_decay = [p for n, p in named if _canonical_name(n) not in self._decay_param_names]
            groups = [{"params": decay, "weight_decay": tc.weight_decay},
                      {"params": no_decay, "weight_decay": 0.0}]
            groups = [g for g in groups if g["params"]]
        if tc.optimizer == "sgd":
            return torch.optim.SGD(groups, lr=tc.learning_rate, momentum=tc.momentum)
        cls = torch.optim.AdamW if tc.optimizer == "adamw" else torch.optim.Adam
        return cls(groups, lr=tc.learning_rate, betas=(tc.beta1, tc.beta2), eps=tc.eps)

    def _build_scheduler(self, lr_scheduler: SchedulerArg):
        if lr_scheduler is not None:
            is_factory = isinstance(lr_scheduler, type) or not hasattr(lr_scheduler, "step")
            return lr_scheduler(self.optimizer) if is_factory else lr_scheduler
        tc = self.config.training
        warmup, total, floor, kind = tc.warmup_steps, max(1, self.total_steps), tc.min_lr_ratio, tc.lr_scheduler

        def lr_lambda(step: int) -> float:  # a closure (not a callable object) so it is not checkpointed
            if step < warmup:
                return (step + 1) / warmup
            if kind == "constant":
                return 1.0
            progress = min(1.0, (step - warmup) / max(1, total - warmup))
            if kind == "linear":
                return 1.0 - (1.0 - floor) * progress
            return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    # ================================================================== forward / backward
    def compute_loss(self, batch: Any) -> torch.Tensor:
        """Run the model on a device-resident batch and return a scalar loss."""
        if isinstance(batch, Mapping):
            outputs = self.model(**batch)
        elif isinstance(batch, (list, tuple)):
            outputs = self.model(*batch)
        else:
            outputs = self.model(batch)
        if isinstance(outputs, torch.Tensor) and outputs.dim() == 0:
            return outputs
        if isinstance(outputs, Mapping) and "loss" in outputs:
            return outputs["loss"]
        if hasattr(outputs, "loss"):
            return outputs.loss
        if isinstance(outputs, (list, tuple)) and len(outputs) >= 2 and isinstance(outputs[1], torch.Tensor):
            return outputs[1]
        raise TypeError("model output must be a scalar loss, contain 'loss', or be a (logits, loss) tuple; "
                        "override Trainer.compute_loss for other conventions")

    def _autocast(self):
        if not self._autocast_enabled:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)

    def _forward_backward(self, batch: Any, sync_gradients: bool) -> torch.Tensor:
        batch = _move_to_device(batch, self.device)
        # Skip the gradient all-reduce on all but the last micro-batch of an accumulation window.
        no_sync = (not sync_gradients) and is_dist() and hasattr(self.model, "no_sync")
        with self.model.no_sync() if no_sync else contextlib.nullcontext():
            with self._autocast():
                loss = self.compute_loss(batch)
            scaled = loss / self.accumulation_steps
            if self.scaler is not None:
                self.scaler.scale(scaled).backward()
            else:
                scaled.backward()
        return loss.detach()

    def _clip_and_get_grad_norm(self) -> float:
        max_norm = self.config.training.max_grad_norm or float("inf")
        if self._is_fsdp:
            norm = self.model.clip_grad_norm_(max_norm)
        else:
            norm = torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.grad is not None],
                                                  max_norm)
        return float(norm)

    def _optimizer_step(self) -> tuple:
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        grad_norm = self._clip_and_get_grad_norm()
        skipped = False
        if self.scaler is not None:
            self.scaler.step(self.optimizer)  # the scaler skips inf/nan steps itself
            self.scaler.update()
        else:
            skipped = self.nonfinite_guard.should_skip(grad_norm, self.global_step + 1)
            if not skipped:
                self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        # Always advance the schedule so that scheduler step == global step (resume invariant).
        with warnings.catch_warnings():
            # torch warns if the scheduler steps before any optimizer.step(), which happens
            # legitimately when the guard (or the fp16 scaler) skips the very first update.
            warnings.filterwarnings("ignore", message=re.escape("Detected call of `lr_scheduler.step()`"))
            self.lr_scheduler.step()
        return grad_norm, skipped

    # ================================================================== main loop
    def train(self) -> TrainResult:
        self._resume()
        self._start_run()
        installed = self.preemption.install() if self.preemption is not None else False
        try:
            status = self._train_loop()
            if self.checkpoint_manager.last_saved_step != self.global_step and (
                status == PREEMPTED or self.config.checkpoint.save_final
            ):
                self.save_checkpoint(blocking=True)
            self.checkpoint_manager.wait_for_pending()
        except BaseException as e:
            self._update_registry(status=FAILED, error=f"{type(e).__name__}: {e}")
            raise
        finally:
            if installed:
                self.preemption.uninstall()

        self._update_registry(status=status, step=self.global_step, finished_at=time.time(),
                              last_checkpoint=self._last_checkpoint)
        if self.metrics_logger is not None:
            self.metrics_logger.log({"status": status}, step=self.global_step, event="end")
        logger.info("Training %s at step %d (epoch %d)", status, self.global_step, self.epoch)
        return TrainResult(status=status, global_step=self.global_step, epoch=self.epoch, run_id=self.run_id,
                           last_checkpoint=self._last_checkpoint, metrics=dict(self.last_metrics))

    def _done(self) -> bool:
        tc = self.config.training
        return ((tc.max_steps is not None and self.global_step >= tc.max_steps)
                or (tc.max_epochs is not None and self.epoch >= tc.max_epochs))

    def _train_loop(self) -> str:
        self.model.train()
        while not self._done():
            self.sampler.set_epoch(self.epoch)
            micro_step = 0
            for batch in self.train_loader:
                micro_step += 1
                boundary = micro_step % self.accumulation_steps == 0
                loss = self._forward_backward(batch, sync_gradients=boundary)
                self._window_loss_sum += loss.float()
                self._window_micro_batches += 1
                self._window_tokens += _count_tokens(batch)
                if not boundary:
                    continue

                grad_norm, skipped = self._optimizer_step()
                self.global_step += 1
                self.sampler.advance(1)
                self._window_steps += 1
                self._last_grad_norm = grad_norm

                if self._after_step():
                    return PREEMPTED
                if self._done():
                    return COMPLETED
            # epoch exhausted
            self.epoch += 1
            self.sampler.set_epoch(self.epoch)
            self.sampler.set_consumed(0)
        return COMPLETED

    def _after_step(self) -> bool:
        """Logging, evaluation and checkpoint decisions. Returns True to stop (preemption)."""
        tc, cc, ft = self.config.training, self.config.checkpoint, self.config.fault_tolerance
        step = self.global_step
        if step % tc.log_interval == 0:
            self._log_window()
        if self.eval_loader is not None and tc.eval_interval and step % tc.eval_interval == 0:
            self.evaluate()

        # Stop / time-based-save decisions must be identical on every rank: agree via one all-reduce.
        stop = time_due = False
        timed = cc.save_interval_minutes is not None
        if (self.preemption is not None or timed) and step % ft.stop_check_interval == 0:
            local_stop = 1.0 if (self.preemption is not None and self.preemption.stop_requested) else 0.0
            elapsed = time.monotonic() - self._last_checkpoint_time
            local_due = 1.0 if timed and elapsed >= cc.save_interval_minutes * 60 else 0.0
            stop_v, due_v = all_reduce_tensor([local_stop, local_due], op="max")
            stop, time_due = stop_v > 0, due_v > 0
        if stop:
            logger.warning("Stop requested (signal %s on some rank): checkpointing at step %d",
                           self.preemption.received_signal or "remote", step)
            return True
        if (cc.save_interval_steps and step % cc.save_interval_steps == 0) or time_due:
            self.save_checkpoint()
        return False

    # ================================================================== evaluation
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        if self.eval_loader is None:
            return {}
        self.model.eval()
        total, count = 0.0, 0
        max_batches = self.config.training.eval_max_batches
        try:
            for i, batch in enumerate(self.eval_loader):
                if max_batches is not None and i >= max_batches:
                    break
                with self._autocast():
                    loss = self.compute_loss(_move_to_device(batch, self.device))
                total += float(loss)
                count += 1
        finally:
            self.model.train()
        total, count = all_reduce_tensor([total, float(count)], op="sum")
        metrics = {"eval/loss": total / max(count, 1.0)}
        if self.is_main_process:
            logger.info("Eval step %d: loss=%.4f", self.global_step, metrics["eval/loss"])
            if self.metrics_logger is not None:
                self.metrics_logger.log(metrics, step=self.global_step)
        self.last_metrics.update(metrics)
        return metrics

    # ================================================================== logging
    def _reset_log_window(self) -> None:
        self._window_loss_sum = torch.zeros((), device=self.device)
        self._window_micro_batches = 0
        self._window_tokens = 0
        self._window_steps = 0
        self._window_start = time.perf_counter()
        self._last_grad_norm = float("nan")

    def _log_window(self) -> None:
        if self._window_steps == 0:
            return
        loss_sum, micro, tokens = all_reduce_tensor(
            [float(self._window_loss_sum), float(self._window_micro_batches), float(self._window_tokens)], op="sum")
        elapsed = time.perf_counter() - self._window_start
        steps = self._window_steps
        metrics = {
            "loss": loss_sum / max(micro, 1.0),
            "lr": self.optimizer.param_groups[0]["lr"],
            "grad_norm": self._last_grad_norm,
            "epoch": self.epoch,
            "step_time_s": elapsed / steps,
            "samples_per_s": self.global_batch * steps / elapsed,
            "skipped_steps": self.nonfinite_guard.total_skipped,
        }
        if tokens:
            metrics["tokens_per_s"] = tokens / elapsed
        if self.checkpoint_manager.last_blocking_seconds is not None:
            metrics["ckpt_block_ms"] = self.checkpoint_manager.last_blocking_seconds * 1e3
        self.last_metrics.update(metrics)
        if self.is_main_process:
            logger.info("step %d | loss %.4f | lr %.2e | grad_norm %.3f | %.0f samples/s",
                        self.global_step, metrics["loss"], metrics["lr"], metrics["grad_norm"],
                        metrics["samples_per_s"])
            if self.metrics_logger is not None:
                self.metrics_logger.log(metrics, step=self.global_step)
            self._update_registry(step=self.global_step, epoch=self.epoch, loss=metrics["loss"], lr=metrics["lr"],
                                  samples_per_s=metrics["samples_per_s"])
        self._reset_log_window()

    def _update_registry(self, **fields: Any) -> None:
        if self.registry is None or self.run_id is None:
            return
        try:
            self.registry.update(self.run_id, **fields)
        except Exception as e:  # tracking must never take training down
            logger.warning("Could not update run registry: %s", e)

    # ================================================================== checkpointing
    def _trainer_state(self) -> Dict[str, Any]:
        return {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "sampler": self.sampler.state_dict(),
            "run_name": self.run_name,
            "nonfinite_guard": self.nonfinite_guard.state_dict(),
        }

    def save_checkpoint(self, blocking: bool = False) -> Optional[str]:
        """Collective: call on every rank. Rank 0 writes; returns the path on rank 0."""
        from flextrain import __version__

        extra = {
            "trainer": self._trainer_state(),
            "rng": all_gather_object(capture_rng_state()),  # one entry per rank
            "grad_scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "global_batch": self.global_batch,
            "config": self.config.to_dict(),
            "flextrain_version": __version__,
        }
        path = self.checkpoint_manager.save(
            self.model, self.optimizer, self.lr_scheduler, step=self.global_step, epoch=self.epoch,
            metrics={k: v for k, v in self.last_metrics.items() if isinstance(v, (int, float))},
            extra=extra, blocking=blocking,
        )
        self._last_checkpoint_time = time.monotonic()
        if path is not None:
            self._last_checkpoint = path
            self._update_registry(last_checkpoint=path, last_checkpoint_step=self.global_step)
        return path

    def _resume(self) -> None:
        cc = self.config.checkpoint
        if cc.resume_path is None and not cc.auto_resume:
            return
        state, path, error = None, None, None
        if self.is_main_process:
            try:
                if cc.resume_path is not None:
                    path, state = cc.resume_path, self.checkpoint_manager.load_state(cc.resume_path)
                else:
                    found = self.checkpoint_manager.load_latest_state()
                    if found is not None:
                        path, state = found
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
        # Every rank resumes from the checkpoint rank 0 validated (or fails together).
        path, error = broadcast_object((path, error))
        if error is not None:
            raise RuntimeError(f"cannot resume from {cc.resume_path}: {error}")
        if path is None:
            logger.info("No checkpoint to resume from; starting fresh")
            return
        if state is None:
            state = self.checkpoint_manager.load_state(path)

        restore_state(state, self.model, self.optimizer, self.lr_scheduler, strict=cc.strict_resume)
        ts = state["trainer"]
        self.global_step = int(ts["global_step"])
        self.epoch = int(ts["epoch"])
        self.sampler.load_state_dict(ts["sampler"])
        self.nonfinite_guard.load_state_dict(ts.get("nonfinite_guard", {}))
        if self.run_name is None:
            self.run_name = ts.get("run_name")
        if self.scaler is not None and state.get("grad_scaler"):
            self.scaler.load_state_dict(state["grad_scaler"])

        rng_states = state.get("rng") or []
        if len(rng_states) == self.world_size:
            restore_rng_state(rng_states[self.rank])
        else:
            # Topology changed: per-rank RNG streams cannot be mapped 1:1, reseed deterministically.
            set_seed(self.config.main.seed + self.rank + 100_003 * self.global_step)
        self.elastic.on_resume({"world_size": state.get("world_size", self.world_size),
                                "global_batch": state.get("global_batch")})
        self.resumed_from = path
        self.checkpoint_manager.last_saved_step = self.global_step
        logger.info("Resumed from %s at step %d (epoch %d, %d samples into the epoch)",
                    path, self.global_step, self.epoch, self.sampler.consumed)

    # ================================================================== run bookkeeping
    def _start_run(self) -> None:
        mc = self.config.main
        if self.run_name is None:
            self.run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_name = broadcast_object(self.run_name)
        self.run_id = make_run_id(mc.experiment_name, self.run_name)
        self.run_dir = self.config.experiment_dir / "runs" / self.run_name
        if mc.log_to_file:
            setup_logging(mc.log_level, self.rank, self.run_dir / "logs" / f"rank{self.rank}.log")
        if not self.is_main_process:
            return
        self.metrics_logger = MetricsLogger(self.run_dir)
        self.config.to_yaml(self.run_dir / "config.yaml")
        self.metrics_logger.log(
            {"world_size": self.world_size, "global_batch": self.global_batch,
             "resumed_from": self.resumed_from, "restart_count": self.elastic.env.restart_count},
            step=self.global_step, event="start")
        existing = self.registry.get(self.run_id) if self.registry is not None else None
        self._update_registry(
            status=RUNNING, experiment=mc.experiment_name, run_name=self.run_name, pid=os.getpid(),
            host=socket.gethostname(), world_size=self.world_size, global_batch=self.global_batch,
            gradient_accumulation_steps=self.accumulation_steps, step=self.global_step, epoch=self.epoch,
            max_steps=self.config.training.max_steps, max_epochs=self.config.training.max_epochs,
            restart_count=self.elastic.env.restart_count, resumed_from=self.resumed_from,
            started_at=(existing or {}).get("started_at") or time.time(), error=None, finished_at=None,
            checkpoint_dir=self.checkpoint_manager.checkpoint_dir,
            metrics_path=str(self.metrics_logger.metrics_file), run_dir=str(self.run_dir),
        )

    def close(self) -> None:
        """Drain checkpoint writes, release resources and tear down the process group we created."""
        self.checkpoint_manager.close()
        teardown_file_logging()
        if self._owns_process_group:
            destroy_distributed()

    def __enter__(self) -> Trainer:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        else:
            self.checkpoint_manager.close(raise_errors=False)
            teardown_file_logging()
            if self._owns_process_group:
                destroy_distributed()


# Backwards-compatible name
DistributedTrainer = Trainer
