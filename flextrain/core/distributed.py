"""Process-group setup and small collective helpers.

Rank and world size come from the standard launcher environment variables
(``RANK``, ``LOCAL_RANK``, ``WORLD_SIZE``, ``MASTER_ADDR``/``MASTER_PORT``) that
``torchrun`` / ``flextrain launch`` export, so the same script runs unchanged on one
device, one node, or many nodes.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, List, Optional

import torch
import torch.distributed as dist

from flextrain.config import DistributedConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DistContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    backend: Optional[str]

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value not in (None, "") else default


def resolve_device(config: DistributedConfig, local_rank: int) -> torch.device:
    want = config.device
    if want == "auto":
        want = "cuda" if torch.cuda.is_available() else "cpu"
    if want == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("distributed.device='cuda' but CUDA is not available")
        index = local_rank % torch.cuda.device_count()
        torch.cuda.set_device(index)
        return torch.device("cuda", index)
    return torch.device("cpu")


def init_distributed(config: Optional[DistributedConfig] = None) -> DistContext:
    """Initialize (or adopt) the default process group and pick this rank's device.

    * If a process group already exists it is reused.
    * If ``WORLD_SIZE`` > 1 a group is created with the configured backend and
      collective timeout, so a dead or hung peer surfaces as an exception (which the
      elastic agent turns into a restart) instead of a job that hangs forever.
    * Otherwise the run is single-process and no group is created.
    """
    config = config or DistributedConfig()
    if dist.is_available() and dist.is_initialized():
        rank, world_size = dist.get_rank(), dist.get_world_size()
        local_rank = _env_int("LOCAL_RANK", rank)
        device = resolve_device(config, local_rank)
        return DistContext(rank, local_rank, world_size, device, dist.get_backend())

    world_size = _env_int("WORLD_SIZE", 1)
    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", 0)
    device = resolve_device(config, local_rank)
    if world_size <= 1:
        return DistContext(0, 0, 1, device, None)

    backend = config.backend
    if backend == "auto":
        backend = "nccl" if device.type == "cuda" else "gloo"
    if backend == "nccl" and device.type != "cuda":
        raise RuntimeError("the nccl backend requires CUDA devices; use backend 'gloo' for CPU")
    timeout = timedelta(minutes=config.timeout_minutes)
    store = _elastic_attempt_store(rank, world_size, timeout)
    if store is not None:
        dist.init_process_group(backend=backend, store=store, rank=rank, world_size=world_size, timeout=timeout)
    else:
        dist.init_process_group(backend=backend, timeout=timeout)
    logger.info("Initialized process group: rank %d/%d (local %d), backend=%s, device=%s",
                rank, world_size, local_rank, backend, device)
    return DistContext(rank, local_rank, world_size, device, backend)


def _elastic_attempt_store(rank: int, world_size: int, timeout: timedelta) -> Optional[dist.Store]:
    """Rendezvous store namespaced by restart attempt when running under torchelastic.

    The elastic agent's TCPStore outlives the worker group. Without a per-attempt namespace a
    restarted rank can read the address its peer published in the *previous* attempt (the
    peer has not overwritten it yet), connect to a dead port, and hang until the timeout.
    Some torch versions add this prefix themselves; doing it here makes restarts reliable
    regardless of version.
    """
    attempt = os.environ.get("TORCHELASTIC_RESTART_COUNT")
    addr, port = os.environ.get("MASTER_ADDR"), os.environ.get("MASTER_PORT")
    if attempt is None or not addr or not port:
        return None
    agent_hosts_store = os.environ.get("TORCHELASTIC_USE_AGENT_STORE") == "True"
    store = dist.TCPStore(addr, int(port), world_size, is_master=(not agent_hosts_store and rank == 0),
                          timeout=timeout, multi_tenant=True)
    return dist.PrefixStore(f"flextrain/attempt_{attempt}", store)


def destroy_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def barrier() -> None:
    if is_dist():
        dist.barrier()


def _collective_device() -> torch.device:
    if is_dist() and dist.get_backend() == "nccl":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def all_reduce_tensor(values: List[float], op: str = "sum") -> List[float]:
    """All-reduce a short list of floats (one collective). No-op when single-process."""
    if not is_dist():
        return list(values)
    tensor = torch.tensor(values, dtype=torch.float64, device=_collective_device())
    reduce_op = {"sum": dist.ReduceOp.SUM, "max": dist.ReduceOp.MAX, "min": dist.ReduceOp.MIN}[op]
    dist.all_reduce(tensor, op=reduce_op)
    return tensor.tolist()


def broadcast_object(obj: Any, src: int = 0) -> Any:
    if not is_dist():
        return obj
    holder = [obj]
    dist.broadcast_object_list(holder, src=src)
    return holder[0]


def all_gather_object(obj: Any) -> List[Any]:
    if not is_dist():
        return [obj]
    out: List[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(out, obj)
    return out
