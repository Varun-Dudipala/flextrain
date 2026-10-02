"""Host-memory snapshots of (possibly GPU-resident) state trees.

Taking the snapshot is the only part of an async checkpoint that blocks training:
tensors are copied device->host so the optimizer can keep mutating the live ones while
a background thread serializes the copy.

Two details keep the blocking time low and the checkpoint small:

* **Buffer reuse.** Host buffers are allocated on the first snapshot and reused by
  later ones (pinned when the source is on CUDA, so copies are async DMA). Allocating
  (and page-faulting, or ``cudaHostAlloc``-ing) a few GB on every save is otherwise
  a large share of the stall.
* **Storage de-duplication.** Tensors that alias the same storage (e.g. tied
  embedding / LM-head weights) are copied once and stay shared, exactly as
  ``torch.save`` would have written them.
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from typing import Any, Dict, Tuple

import torch

_MemoKey = Tuple[Any, ...]


def _memo_key(t: torch.Tensor) -> _MemoKey:
    return (t.device, t.untyped_storage().data_ptr(), t.storage_offset(), tuple(t.shape), t.stride(), t.dtype)


class SnapshotBuffers:
    """A reusable set of host buffers, keyed by the tensor's path in the state tree."""

    def __init__(self, pin_memory: bool = False, reuse: bool = True):
        self.pin_memory = pin_memory and torch.cuda.is_available()
        self.reuse = reuse
        self._buffers: Dict[str, torch.Tensor] = {}

    @property
    def nbytes(self) -> int:
        return sum(b.numel() * b.element_size() for b in self._buffers.values())

    def release(self) -> None:
        self._buffers.clear()

    def snapshot(self, obj: Any) -> Any:
        memo: Dict[_MemoKey, torch.Tensor] = {}
        used_cuda = [False]
        out = self._copy(obj, "", memo, used_cuda)
        if used_cuda[0]:
            torch.cuda.synchronize()  # non_blocking D2H copies must land before the writer reads them
        return out

    def _buffer_for(self, key: str, t: torch.Tensor) -> torch.Tensor:
        buf = self._buffers.get(key) if self.reuse else None
        if buf is None or buf.shape != t.shape or buf.dtype != t.dtype:
            buf = torch.empty(t.shape, dtype=t.dtype, device="cpu", pin_memory=self.pin_memory and t.is_cuda)
            if self.reuse:
                self._buffers[key] = buf
        return buf

    def _copy(self, obj: Any, key: str, memo: Dict[_MemoKey, torch.Tensor], used_cuda: list) -> Any:
        if isinstance(obj, torch.Tensor):
            if obj.layout != torch.strided:
                return obj.detach().to("cpu", copy=True)
            mkey = _memo_key(obj)
            if mkey in memo:
                return memo[mkey]
            src = obj.detach()
            buf = self._buffer_for(key, src)
            buf.copy_(src, non_blocking=src.is_cuda)
            used_cuda[0] |= src.is_cuda
            memo[mkey] = buf
            return buf
        if isinstance(obj, dict):
            items = ((k, self._copy(v, f"{key}/{k}", memo, used_cuda)) for k, v in obj.items())
            return OrderedDict(items) if isinstance(obj, OrderedDict) else dict(items)
        if isinstance(obj, (list, tuple)):
            items = [self._copy(v, f"{key}/{i}", memo, used_cuda) for i, v in enumerate(obj)]
            if hasattr(obj, "_fields"):  # namedtuple
                return type(obj)(*items)
            return type(obj)(items) if isinstance(obj, tuple) else items
        return copy.deepcopy(obj)


def snapshot_to_cpu(obj: Any, pin_memory: bool = False) -> Any:
    """One-off snapshot without buffer reuse."""
    return SnapshotBuffers(pin_memory=pin_memory, reuse=False).snapshot(obj)
