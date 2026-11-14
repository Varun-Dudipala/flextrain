"""Randomized fault injection as a regression test: a few fixed-seed trials of the chaos harness.

Each trial SIGKILLs random ranks while the job trains and requires the result to be bitwise
identical to an uninterrupted run (see ``tests/integration/chaos.py`` for everything a trial
checks). The seeds are chosen to cover a kill of either rank, a kill before the first checkpoint
exists, and two kills in one job. For a real campaign run
``python -m tests.integration.chaos --trials 50``.
"""

from __future__ import annotations

import random

import pytest
import torch

from tests.integration import chaos

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
                       reason="requires torch.distributed with gloo"),
]

MAX_KILLS = 2
# seed -> the kill plan it draws: [(rank, step), ...]
SEEDS = {
    4: [(0, 20)],  # rank 0 (the rank that writes checkpoints), mid-run
    42: [(1, 2)],  # before the first checkpoint: the restart has nothing to resume from
    9: [(1, 24), (0, 40)],  # two kills, one per rank; the second hits the restarted job
}


@pytest.fixture(scope="module")
def reference(tmp_path_factory) -> str:
    return chaos.reference_digest(tmp_path_factory.mktemp("chaos-reference"))


def test_seeds_draw_the_documented_kill_plans():
    for seed, plan in SEEDS.items():
        drawn = chaos.plan_kills(random.Random(seed), MAX_KILLS)
        assert [(k.rank, k.planned_step) for k in drawn] == plan


@pytest.mark.parametrize("seed", list(SEEDS))
def test_random_kills_never_change_the_result(seed, reference, tmp_path):
    trial = chaos.run_trial(seed, reference, max_kills=MAX_KILLS, workdir=tmp_path, keep_failed=True)
    assert trial.passed, chaos.describe(trial)
    assert trial.delivered == len(SEEDS[seed]) and trial.restarts == trial.delivered, chaos.describe(trial)
