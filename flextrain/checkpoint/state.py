"""Wrapper-agnostic capture / restore of model and optimizer state.

Uses ``torch.distributed.checkpoint.state_dict`` so that plain modules, DDP and FSDP all
produce the same *full*, fully-qualified-name keyed state dict. Consequences:

* a checkpoint written by a DDP job can be resumed under FSDP (or on a single device),
* the ``module.`` / ``_orig_mod.`` wrapper prefixes never leak into checkpoints,
* for FSDP the gather is a collective: **every rank must call** ``capture_*``; only
  rank 0 receives the full state (others get an empty dict) and only rank 0 writes.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)

FORMAT_VERSION = 1


def is_sharded(model: nn.Module) -> bool:
    """True if any part of ``model`` is FSDP-wrapped (parameters are sharded across ranks)."""
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    except ImportError:  # pragma: no cover
        return False
    return any(isinstance(m, FSDP) for m in model.modules())


def _save_options(model: nn.Module) -> StateDictOptions:
    # For sharded models, cpu_offload + full_state_dict gathers to rank 0 only (no N full copies).
    return StateDictOptions(full_state_dict=True, cpu_offload=is_sharded(model))


def _load_options(strict: bool = True) -> StateDictOptions:
    # Every rank passes the full (CPU) state dict and keeps only its shard. cpu_offload must stay
    # off here: torch maps it to rank0_only, which expects non-zero ranks to pass nothing.
    return StateDictOptions(full_state_dict=True, cpu_offload=False, strict=strict)


def capture_model_state(model: nn.Module) -> Dict[str, Any]:
    return get_model_state_dict(model, options=_save_options(model))


def capture_optimizer_state(model: nn.Module, optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
    return get_optimizer_state_dict(model, optimizer, options=_save_options(model))


def restore_model_state(model: nn.Module, state: Dict[str, Any], strict: bool = True) -> Any:
    return set_model_state_dict(model, state, options=_load_options(strict))


def restore_optimizer_state(model: nn.Module, optimizer: torch.optim.Optimizer, state: Dict[str, Any]) -> None:
    set_optimizer_state_dict(model, optimizer, state, options=_load_options())


def capture_state(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr_scheduler: Optional[Any] = None,
) -> Dict[str, Any]:
    """Model / optimizer / scheduler state in checkpoint layout (collective under FSDP)."""
    return {
        "format_version": FORMAT_VERSION,
        "model": capture_model_state(model),
        "optimizer": capture_optimizer_state(model, optimizer) if optimizer is not None else None,
        "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
    }


def restore_state(
    state: Dict[str, Any],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr_scheduler: Optional[Any] = None,
    strict: bool = True,
) -> None:
    restore_model_state(model, state["model"], strict=strict)
    if optimizer is not None and state.get("optimizer"):
        restore_optimizer_state(model, optimizer, state["optimizer"])
    if lr_scheduler is not None and state.get("lr_scheduler"):
        lr_scheduler.load_state_dict(state["lr_scheduler"])


# ---------------------------------------------------------------------------- RNG state
def capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {"python": random.getstate(), "torch": torch.get_rng_state()}
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:  # pragma: no cover
        pass
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        state["cuda"] = torch.cuda.get_rng_state()
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if "numpy" in state:
        try:
            import numpy as np

            np.random.set_state(state["numpy"])
        except ImportError:  # pragma: no cover
            pass
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda"])


def validate_checkpoint(state: Any) -> List[str]:
    """Return a list of problems (empty if ``state`` looks like a usable checkpoint)."""
    if not isinstance(state, dict):
        return [f"expected a dict, got {type(state).__name__}"]
    problems = []
    version = state.get("format_version")
    if version is None:
        problems.append("missing format_version (not a FlexTrain >= 0.2 checkpoint)")
    elif version > FORMAT_VERSION:
        problems.append(f"format_version {version} is newer than supported ({FORMAT_VERSION})")
    if not isinstance(state.get("model"), dict):
        problems.append("missing model state")
    if not isinstance(state.get("step"), int):
        problems.append("missing step")
    return problems
