"""Distributed data loading utilities."""

from typing import Iterator, Optional
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler


class ElasticDistributedSampler(DistributedSampler):
    """Sampler that handles elastic scaling."""

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ):
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )
        self._current_index = 0

    def set_progress(self, batch_idx: int, batch_size: int) -> None:
        """Set progress for resuming after scaling event."""
        self._current_index = batch_idx * batch_size

    def __iter__(self) -> Iterator:
        indices = list(super().__iter__())
        if self._current_index > 0:
            indices = indices[self._current_index:]
            self._current_index = 0
        return iter(indices)


def create_distributed_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: bool = True,
    drop_last: bool = True,
    world_size: int = 1,
    rank: int = 0,
    elastic: bool = False,
) -> DataLoader:
    """Create a distributed data loader."""

    sampler = None
    if world_size > 1:
        if elastic:
            sampler = ElasticDistributedSampler(dataset, shuffle=shuffle)
        else:
            sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
