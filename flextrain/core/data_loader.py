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

Which samples are drawn is only half of reproducible data: random augmentation is the other
half. DataLoader workers each carry their own RNG stream whose state depends on how many
workers there are and on everything they have produced since the epoch began, so it cannot
be restored mid-epoch. ``SeededDataset`` removes the dependency instead: it seeds the RNGs
from ``(seed, epoch, sample index)`` before each sample is produced, making a sample's
randomness a pure function of *which* sample it is - not of the worker that produced it,
the number of workers or ranks, or where in the epoch a resumed run started.
"""

from __future__ import annotations

import contextlib
import hashlib
import random
import struct
from typing import Any, Callable, Dict, Iterator, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Sampler, default_collate, get_worker_info

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

_SAMPLE_STREAM, _COLLATE_STREAM = 0, 1


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
        return {"epoch": self.epoch, "consumed": self.consumed, "seed": self.seed, "shuffle": self.shuffle,
                "dataset_len": self.dataset_len}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if state.get("seed", self.seed) != self.seed or state.get("shuffle", self.shuffle) != self.shuffle:
            raise ValueError("cannot resume: sampler seed/shuffle differ from the checkpoint "
                             "(the data order would change)")
        if state.get("dataset_len", self.dataset_len) != self.dataset_len:
            raise ValueError(f"cannot resume: the dataset has {self.dataset_len} samples but the checkpoint was "
                             f"written with {state['dataset_len']} (the position within the epoch would point at "
                             f"different samples)")
        self.epoch = int(state["epoch"])
        self.consumed = int(state["consumed"])


def derive_seed(seed: int, epoch: int, index: int, stream: int = _SAMPLE_STREAM) -> int:
    """A well-mixed 64-bit seed for one (epoch, sample); identical in every process."""
    digest = hashlib.blake2b(struct.pack("<4q", seed, epoch, index, stream), digest_size=8).digest()
    return int.from_bytes(digest, "little")


@contextlib.contextmanager
def seeded_rng(seed: int) -> Iterator[None]:
    """Run a block with the Python, NumPy and torch (CPU) global RNGs seeded from ``seed``.

    In a DataLoader worker the RNGs exist only to produce samples, so they are simply
    reseeded. With ``num_workers=0`` the block runs in the training process, whose RNG
    stream also drives the model (dropout, ...): there the previous states are restored
    afterwards, so loading data never perturbs the model's stream. Only the CPU generator is
    touched (``torch.manual_seed`` would reseed the CUDA generators as well).
    """
    in_worker = get_worker_info() is not None
    if not in_worker:
        saved = (random.getstate(), torch.get_rng_state(), np.random.get_state() if np is not None else None)
    random.seed(seed)
    torch.default_generator.manual_seed(seed)
    if np is not None:
        np.random.seed(seed % 2**32)
    try:
        yield
    finally:
        if not in_worker:
            random.setstate(saved[0])
            torch.set_rng_state(saved[1])
            if np is not None:
                np.random.set_state(saved[2])


class SeededDataset(Dataset):
    """Makes ``dataset``'s randomness a pure function of ``(seed, epoch, sample index)``.

    Items are returned as ``(index, item)`` so that ``SeededCollate`` can seed batch-level
    randomness (e.g. dynamic masking in a collate function) the same way. The epoch is read
    from the sampler, which is the single source of truth for the data position; workers
    receive a copy of it when an epoch's iterator is created.
    """

    def __init__(self, dataset: Dataset, seed: int, sampler: ResumableDistributedSampler):
        self.dataset = dataset
        self.seed = seed
        self.sampler = sampler

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[int, Any]:
        with seeded_rng(derive_seed(self.seed, self.sampler.epoch, index)):
            return index, self.dataset[index]


class SeededCollate:
    """Collates the items of ``SeededDataset``, seeding the RNGs from the batch's first sample."""

    def __init__(self, collate_fn: Optional[Callable], seed: int, sampler: ResumableDistributedSampler):
        self.collate_fn = collate_fn or default_collate
        self.seed = seed
        self.sampler = sampler

    def __call__(self, indexed_items: Sequence[Tuple[int, Any]]) -> Any:
        indices, items = zip(*indexed_items, strict=True)
        with seeded_rng(derive_seed(self.seed, self.sampler.epoch, indices[0], _COLLATE_STREAM)):
            return self.collate_fn(list(items))


def build_dataloader(
    dataset: Dataset,
    sampler: Sampler,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    collate_fn: Optional[Callable] = None,
    seed: int = 0,
    sample_seed: Optional[int] = None,
) -> DataLoader:
    """Build the training loader.

    ``sample_seed`` (the run seed, the same on every rank) enables per-sample seeding of
    dataset and collate randomness; ``None`` leaves both to the ambient RNG streams.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    if sample_seed is not None:
        dataset = SeededDataset(dataset, sample_seed, sampler)
        collate_fn = SeededCollate(collate_fn, sample_seed, sampler)
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
