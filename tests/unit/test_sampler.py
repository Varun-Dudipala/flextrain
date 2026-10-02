"""ResumableDistributedSampler: determinism, sharding, mid-epoch resume, elastic re-sharding."""

import pytest
import torch

from flextrain.core.data_loader import ResumableDistributedSampler, build_dataloader


def per_step_global(n, world, per_rank_step, steps, consumed=0, seed=0, epoch=0):
    """Simulate `steps` optimizer steps on `world` ranks; return the global sample set of each step."""
    samplers = [ResumableDistributedSampler(n, world, r, seed=seed, samples_per_step=per_rank_step)
                for r in range(world)]
    rank_lists = []
    for s in samplers:
        s.set_epoch(epoch)
        s.set_consumed(consumed)
        rank_lists.append(list(s))
    out = []
    for k in range(steps):
        block = []
        for lst in rank_lists:
            block += lst[k * per_rank_step : (k + 1) * per_rank_step]
        out.append(sorted(block))
    return out


def test_ranks_partition_the_epoch_without_overlap():
    n, world = 100, 4
    shards = [list(ResumableDistributedSampler(n, world, r, samples_per_step=2)) for r in range(world)]
    flat = [i for shard in shards for i in shard]
    assert len(flat) == len(set(flat)) == 96  # 100 // (2*4) = 12 whole steps -> 96 samples
    assert all(len(s) == 24 for s in shards)


def test_same_seed_and_epoch_is_deterministic_and_epochs_differ():
    a = ResumableDistributedSampler(50, seed=3)
    b = ResumableDistributedSampler(50, seed=3)
    assert list(a) == list(b)
    b.set_epoch(1)
    assert list(a) != list(b)


def test_no_shuffle_is_sequential():
    s = ResumableDistributedSampler(10, 2, 1, shuffle=False)
    assert list(s) == [1, 3, 5, 7, 9]


def test_each_step_is_a_contiguous_block_of_the_permutation():
    n, world, per_rank = 64, 4, 2
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0)).tolist()
    steps = per_step_global(n, world, per_rank, steps=8)
    for k, block in enumerate(steps):
        assert block == sorted(perm[k * 8 : (k + 1) * 8])


def test_resume_mid_epoch_continues_exactly():
    n, world, per_rank = 64, 2, 4
    full = per_step_global(n, world, per_rank, steps=8)
    resumed = per_step_global(n, world, per_rank, steps=5, consumed=3 * world * per_rank)
    assert resumed == full[3:]


@pytest.mark.parametrize("old_world,new_world", [(4, 2), (2, 4), (4, 1), (1, 3)])
def test_elastic_resize_neither_repeats_nor_skips(old_world, new_world):
    """Global batch fixed at 12: consume 2 steps on old_world ranks, then finish on new_world ranks."""
    n, global_batch, seed = 120, 12, 7
    before = per_step_global(n, old_world, global_batch // old_world, steps=2, seed=seed)
    consumed = 2 * global_batch
    after_steps = (n - consumed) // global_batch
    after = per_step_global(n, new_world, global_batch // new_world, steps=after_steps, consumed=consumed, seed=seed)
    seen = [i for block in before + after for i in block]
    assert sorted(seen) == list(range(n))  # every sample exactly once
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    assert after[0] == sorted(perm[consumed : consumed + global_batch])


def test_len_and_remaining_steps():
    s = ResumableDistributedSampler(100, num_replicas=2, rank=0, samples_per_step=5)
    assert s.steps_per_epoch == 10 and len(s) == 50
    s.advance(3)
    assert s.consumed == 30 and s.remaining_steps() == 7 and len(s) == 35
    s.set_consumed(100)
    assert len(s) == 0 and list(s) == []


def test_state_dict_round_trip_and_guard():
    s = ResumableDistributedSampler(100, seed=1)
    s.set_epoch(3)
    s.advance(10)
    t = ResumableDistributedSampler(100, seed=1)
    t.load_state_dict(s.state_dict())
    assert (t.epoch, t.consumed) == (3, 10) and list(t) == list(s)
    with pytest.raises(ValueError, match="seed"):
        ResumableDistributedSampler(100, seed=2).load_state_dict(s.state_dict())


@pytest.mark.parametrize("kwargs", [{"num_replicas": 0}, {"num_replicas": 2, "rank": 2}, {"samples_per_step": 0}])
def test_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        ResumableDistributedSampler(10, **kwargs)


def test_dataloader_yields_whole_batches_from_sampler():
    data = list(range(20))
    sampler = ResumableDistributedSampler(len(data), shuffle=False, samples_per_step=4)
    loader = build_dataloader(data, sampler, batch_size=2)
    batches = [b.tolist() for b in loader]
    assert batches == [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19]]
