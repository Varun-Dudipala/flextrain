"""CheckpointManager: naming, retention, corruption fallback, resume, multi-rank behaviour."""

import logging
import threading
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from flextrain.checkpoint import CheckpointManager, CheckpointSaveError, LocalStorage, validate_checkpoint
from flextrain.config import CheckpointConfig


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def manager(tmp_path, **kwargs) -> CheckpointManager:
    storage = kwargs.pop("storage", None)
    rank = kwargs.pop("rank", 0)
    return CheckpointManager(CheckpointConfig(checkpoint_dir=str(tmp_path), **kwargs), rank=rank, storage=storage)


def names(tmp_path):
    return sorted(p.name for p in Path(tmp_path).glob("*.pt"))


def train_a_bit(model, optimizer, steps=3):
    for _ in range(steps):
        model(torch.randn(4, 10)).pow(2).mean().backward()
        optimizer.step()
        optimizer.zero_grad()


@pytest.mark.parametrize("async_save", [False, True])
def test_save_and_load_round_trip(tmp_path, async_save):
    model = SimpleModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
    train_a_bit(model, optimizer)
    scheduler.step()
    expected = {k: v.clone() for k, v in model.state_dict().items()}

    with manager(tmp_path, async_save=async_save) as mgr:
        path = mgr.save(model, optimizer, scheduler, step=100, epoch=5, metrics={"loss": 0.5}, blocking=True)
        assert Path(path).name == "checkpoint_step00000100.pt"

        fresh = SimpleModel()
        fresh_opt = torch.optim.AdamW(fresh.parameters(), lr=1e-3)
        fresh_sched = torch.optim.lr_scheduler.StepLR(fresh_opt, step_size=2)
        meta = mgr.load(fresh, fresh_opt, fresh_sched)

    assert meta["step"] == 100 and meta["epoch"] == 5 and meta["metrics"] == {"loss": 0.5}
    for k, v in fresh.state_dict().items():
        assert torch.equal(v, expected[k])
    assert fresh_sched.last_epoch == scheduler.last_epoch
    # Optimizer moments restored: one more identical step keeps both models bitwise identical.
    x = torch.randn(4, 10)
    for m, o in ((model, optimizer), (fresh, fresh_opt)):
        m(x).pow(2).mean().backward()
        o.step()
        o.zero_grad()
    assert all(torch.equal(p, q) for p, q in zip(model.parameters(), fresh.parameters(), strict=True))


def test_latest_is_by_step_not_mtime(tmp_path):
    model = SimpleModel()
    with manager(tmp_path, async_save=False) as mgr:
        mgr.save(model, step=500)
        mgr.save(model, step=100)  # written later, but older in training progress
        assert Path(mgr.latest_checkpoint()).name == "checkpoint_step00000500.pt"
        assert [c.step for c in mgr.list_checkpoints()] == [100, 500]


def test_retention_keeps_newest_n(tmp_path):
    model = SimpleModel()
    with manager(tmp_path, async_save=True, keep_last_n=2, max_pending_saves=3) as mgr:
        for step in range(1, 7):
            mgr.save(model, step=step)
    assert names(tmp_path) == ["checkpoint_step00000005.pt", "checkpoint_step00000006.pt"]


def test_retention_survives_restarts(tmp_path):
    model = SimpleModel()
    for run in range(3):
        with manager(tmp_path, async_save=False, keep_last_n=2) as mgr:
            for step in range(2):
                mgr.save(model, step=run * 10 + step)
    assert names(tmp_path) == ["checkpoint_step00000020.pt", "checkpoint_step00000021.pt"]


def test_failed_async_save_never_prunes_older_checkpoints(tmp_path):
    class FailOnStep3(LocalStorage):
        def save(self, state, path):
            if state["step"] == 3:
                raise OSError("disk full")
            return super().save(state, path)

    model = SimpleModel()
    mgr = manager(tmp_path, async_save=True, keep_last_n=2, storage=FailOnStep3())
    mgr.write_retries = 0
    mgr._writer.write_retries = 0
    mgr.save(model, step=1)
    mgr.save(model, step=2)
    mgr.save(model, step=3)
    with pytest.raises(CheckpointSaveError):
        mgr.wait_for_pending()
    assert names(tmp_path) == ["checkpoint_step00000001.pt", "checkpoint_step00000002.pt"]
    mgr.close()


def test_corrupted_latest_falls_back_to_previous(tmp_path, caplog):
    model = SimpleModel()
    with manager(tmp_path, async_save=False) as mgr:
        mgr.save(model, step=1)
        latest = mgr.save(model, step=2)
        Path(latest).write_bytes(Path(latest).read_bytes()[:200])  # torn file (e.g. bit rot / bad copy)
        with caplog.at_level(logging.WARNING):
            meta = mgr.try_load_latest(model)
    assert meta["step"] == 1
    assert "Skipping unusable checkpoint" in caplog.text


def test_non_flextrain_file_is_skipped(tmp_path):
    model = SimpleModel()
    with manager(tmp_path, async_save=False) as mgr:
        mgr.save(model, step=1)
        torch.save({"random": "dict"}, tmp_path / "checkpoint_step00000009.pt")
        assert mgr.load_latest_state()[1]["step"] == 1


def test_try_load_latest_empty(tmp_path):
    with manager(tmp_path, async_save=False) as mgr:
        assert mgr.try_load_latest(SimpleModel()) is None
        with pytest.raises(FileNotFoundError):
            mgr.load(SimpleModel())


def test_stale_temp_files_are_cleaned_up(tmp_path):
    (tmp_path / ".checkpoint_step00000007.pt.deadbeef.tmp").write_bytes(b"partial")
    (tmp_path / "unrelated.tmp").write_bytes(b"keep me")
    manager(tmp_path, async_save=False).close()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["unrelated.tmp"]


def test_non_main_rank_does_not_write(tmp_path):
    with manager(tmp_path, rank=1) as mgr:
        assert mgr.save(SimpleModel(), step=1) is None
        assert mgr.last_saved_step == 1  # kept in sync for collective decisions
    assert names(tmp_path) == []


def test_async_save_blocks_only_for_snapshot(tmp_path):
    gate = threading.Event()

    class SlowStorage(LocalStorage):
        def save(self, state, path):
            gate.wait(10)
            return super().save(state, path)

    mgr = manager(tmp_path, async_save=True, storage=SlowStorage())
    path = mgr.save(SimpleModel(), step=1)
    assert mgr.pending_saves() == 1 and not Path(path).exists()
    gate.set()
    mgr.wait_for_pending()
    assert Path(path).exists() and mgr.pending_saves() == 0
    mgr.close()


def test_save_optimizer_false_omits_optimizer(tmp_path):
    model = SimpleModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    with manager(tmp_path, async_save=False, save_optimizer=False) as mgr:
        path = mgr.save(model, opt, step=1)
    assert torch.load(path, weights_only=False)["optimizer"] is None


def test_checkpoint_is_wrapper_agnostic(tmp_path):
    """No 'module.' prefixes: a checkpoint loads into the bare module regardless of wrapping."""
    model = SimpleModel()
    with manager(tmp_path, async_save=False) as mgr:
        path = mgr.save(model, step=1)
    state = torch.load(path, weights_only=False)
    assert set(state["model"]) == set(SimpleModel().state_dict())
    assert validate_checkpoint(state) == []


def test_validate_checkpoint_reports_problems():
    assert validate_checkpoint("nope")
    assert any("format_version" in p for p in validate_checkpoint({"model": {}, "step": 1}))
    assert any("newer" in p for p in validate_checkpoint({"format_version": 99, "model": {}, "step": 1}))
