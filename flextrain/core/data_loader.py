"""Resumable, elastic-aware distributed data loading.

``torch.utils.data.DistributedSampler`` can only restart an epoch from the beginning,
and its per-rank shards depend on the world size. After a crash that means repeated
data; after an elastic resize the progress cannot even be expressed.

``ResumableDistributedSampler`` tracks progress as a single *global* number - samples
of this epoch's permutation already consumed - which is topology independent:

1. each epoch has one deterministic global permutation (``seed + epoch``);
2. the consumed prefix is skipped and the remainder is split across the *current*
   ranks so that optimizer step ``k`` uses the contiguous block
   ``remaining[k * G : (k + 1) * G]`` (``G`` = global batch);
3. hence after any number of steps the consumed samples are exactly a prefix of the
   permutation, regardless of how many ranks consumed them.

Resuming with a different world size therefore continues exactly where the previous
topology stopped - no sample is repeated or skipped (except the epoch's tail that does
not fill a whole global batch, which is dropped as with ``drop_last``).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterator, Optional

import torch
from torch.utils.data import DataLoader, Dataset, Sampler


class ResumableDistributedSampler(Sampler[int]):
    """Distributed sampler with a global, resumable position within the epoch.

    Args:
        dataset_len: number of samples in the dataset.
        num_replicas / rank: current topology.
        shuffle / seed: one permutation per epoch, generated from ``seed + epoch``.
        samples_per_step: per-rank samples per optimizer step
            (``micro_batch * grad_accumulation``). Epochs contain only whole steps.
    """

    def __init__(
        self,
        dataset_len: int,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        seed: int = 0,
        samples_per_step: int = 1,
    ):
        if num_replicas < 1 or not 0 <= rank < num_replicas:
            raise ValueError(f"invalid rank {rank} for num_replicas {num_replicas}")
        if samples_per_step < 1:
            raise ValueError("samples_per_step must be >= 1")
        self.dataset_len = dataset_len
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.samples_per_step = samples_per_step
        self.epoch = 0
        self.consumed = 0  # global samples of this epoch's permutation already used

    @property
    def global_step_size(self) -> int:
        return self.samples_per_step * self.num_replicas

    @property
    def steps_per_epoch(self) -> int:
        """Optimizer steps in a full epoch at the current topology."""
        return self.dataset_len // self.global_step_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def set_consumed(self, consumed: int) -> None:
        if consumed < 0:
            raise ValueError("consumed must be >= 0")
        self.consumed = consumed

    def advance(self, num_steps: int = 1) -> None:
        """Record that ``num_steps`` optimizer steps' worth of samples were consumed."""
        self.consumed += num_steps * self.global_step_size

    def remaining_steps(self) -> int:
        return max(0, self.dataset_len - self.consumed) // self.global_step_size

    def _permutation(self) -> torch.Tensor:
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            return torch.randperm(self.dataset_len, generator=g)
        return torch.arange(self.dataset_len)

    def __iter__(self) -> Iterator[int]:
        n_steps = self.remaining_steps()
        remaining = self._permutation()[self.consumed : self.consumed + n_steps * self.global_step_size]
        # Interleaved split: per-rank sample j of step k is remaining[(k*S + j) * W + rank],
        # so step k across all ranks covers remaining[k*S*W : (k+1)*S*W] (a contiguous block).
        return iter(remaining[self.rank :: self.num_replicas].tolist())

    def __len__(self) -> int:
        return self.remaining_steps() * self.samples_per_step

    def state_dict(self) -> Dict[str, Any]:
        return {"epoch": self.epoch, "consumed": self.consumed, "seed": self.seed, "shuffle": self.shuffle}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if state.get("seed", self.seed) != self.seed or state.get("shuffle", self.shuffle) != self.shuffle:
            raise ValueError("cannot resume: sampler seed/shuffle differ from the checkpoint "
                             "(the data order would change)")
        self.epoch = int(state["epoch"])
        self.consumed = int(state["consumed"])


def build_dataloader(
    dataset: Dataset,
    sampler: Sampler,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    collate_fn: Optional[Callable] = None,
    seed: int = 0,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=collate_fn,
        generator=generator,
        persistent_workers=False,  # the sampler is re-positioned between epochs / resumes
    )
