"""CheckpointManager: naming, retention, corruption fallback, resume, multi-rank behaviour."""

import logging
import threading
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from flextrain.checkpoint import (
    CheckpointConflictError,
    CheckpointManager,
    CheckpointSaveError,
    LocalStorage,
    NoUsableCheckpointError,
    validate_checkpoint,
)
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


def test_bit_rot_in_latest_falls_back_to_previous(tmp_path, caplog):
    """A flipped byte in the weights still unpickles; only the digest catches it."""
    model = SimpleModel()
    with manager(tmp_path, async_save=False) as mgr:
        mgr.save(model, step=1)
        latest = Path(mgr.save(model, step=2))
        raw = bytearray(latest.read_bytes())
        offset = bytes(raw).find(model.fc1.weight.detach().numpy().tobytes())
        assert offset > 0
        raw[offset] ^= 0xFF
        latest.write_bytes(raw)
        with caplog.at_level(logging.WARNING):
            meta = mgr.try_load_latest(model)
    assert meta["step"] == 1
    assert "checksum mismatch" in caplog.text


def test_truncated_checkpoint_without_a_digest_still_falls_back(tmp_path):
    """The path that existed before digests: nothing to compare against, the bytes just do not load."""
    model = SimpleModel()
    with manager(tmp_path, async_save=False) as mgr:
        mgr.save(model, step=1)
        latest = Path(mgr.save(model, step=2))
        Path(str(latest) + ".sha256").unlink()
        latest.write_bytes(latest.read_bytes()[:200])
        assert mgr.try_load_latest(model)["step"] == 1


def test_pruning_removes_digest_files_too(tmp_path):
    model = SimpleModel()
    with manager(tmp_path, async_save=False, keep_last_n=1) as mgr:
        for step in range(1, 4):
            mgr.save(model, step=step)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["checkpoint_step00000003.pt",
                                                          "checkpoint_step00000003.pt.sha256", "manifest.json"]


def test_checksum_config_disables_digests(tmp_path):
    with manager(tmp_path, async_save=False, checksum=False) as mgr:
        mgr.save(SimpleModel(), step=1)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["checkpoint_step00000001.pt", "manifest.json"]


def corrupt(path):
    Path(path).write_bytes(Path(path).read_bytes()[:200])


class TestLineage:
    """A directory plus a step number is not an identity; the manifest's lineage is."""

    def test_new_lineage_is_not_crowded_out_by_an_older_runs_higher_steps(self, tmp_path):
        """Step-based retention used to delete a new run's checkpoints the moment they were written,
        because an earlier run in the same directory had got further."""
        model = SimpleModel()
        with manager(tmp_path, async_save=False, keep_last_n=3) as old:
            for step in (16, 18, 20):
                old.save(model, step=step)
        with manager(tmp_path, async_save=False, keep_last_n=1) as new:
            new.open_lineage()
            new.save(model, step=2)
            path = new.save(model, step=4)
            assert Path(path).exists() and new.latest_checkpoint() == path
            assert [c.step for c in new.committed_checkpoints()] == [4]
        # its own step 2 was pruned; the earlier run's files were never touched
        assert names(tmp_path) == [f"checkpoint_step{s:08d}.pt" for s in (4, 16, 18, 20)]

    def test_continuing_from_an_older_checkpoint_drops_the_abandoned_future(self, tmp_path):
        """Fall back from a corrupted step 20 to step 10, then save step 13. Keeping the 'highest
        step' would keep the corrupted 20 and prune 13, leaving nothing usable."""
        model = SimpleModel()
        with manager(tmp_path, async_save=False, keep_last_n=2) as first:
            first.save(model, step=10)
            corrupt(first.save(model, step=20))
        with manager(tmp_path, async_save=False, keep_last_n=1) as second:
            assert second.load_latest_state()[1]["step"] == 10
            second.save(model, step=13)
            assert [c.step for c in second.committed_checkpoints()] == [13]
        with manager(tmp_path, async_save=False) as third:
            assert third.load_latest_state()[1]["step"] == 13

    def test_a_file_is_not_a_checkpoint_until_it_is_committed(self, tmp_path):
        """Writer dies after the checkpoint file is durable but before the manifest names it."""
        model = SimpleModel()

        class DiesBeforeCommit(LocalStorage):
            def write_bytes(self, path, data):
                if Path(path).name == "manifest.json" and b'"step": 2' in data:
                    raise OSError("killed before the commit")
                super().write_bytes(path, data)

        mgr = manager(tmp_path, async_save=False, storage=DiesBeforeCommit())
        mgr.write_retries = 0
        mgr.save(model, step=1)
        with pytest.raises(OSError, match="killed before the commit"):
            mgr.save(model, step=2)
        assert (tmp_path / "checkpoint_step00000002.pt").exists()
        with manager(tmp_path, async_save=False) as after:
            assert [c.step for c in after.committed_checkpoints()] == [1]
            assert after.load_latest_state()[1]["step"] == 1

    @pytest.mark.parametrize("async_save", [False, True])
    def test_a_second_writer_fences_the_first_one_off(self, tmp_path, async_save):
        model = SimpleModel()
        a = manager(tmp_path, async_save=async_save)
        b = manager(tmp_path, async_save=False)
        a.save(model, step=1, blocking=True)
        b.save(model, step=2)  # takes the directory over
        with pytest.raises((CheckpointConflictError, CheckpointSaveError), match="another process"):
            a.save(model, step=3, blocking=True)
        assert [c.step for c in b.committed_checkpoints()] == [1, 2]
        a.close(raise_errors=False)
        b.close()

    def test_directory_without_a_manifest_is_adopted(self, tmp_path):
        """Checkpoints written before manifests existed keep working, including retention."""
        model = SimpleModel()
        with manager(tmp_path, async_save=False) as old:
            for step in (1, 2, 3):
                old.save(model, step=step)
        (tmp_path / "manifest.json").unlink()
        with manager(tmp_path, async_save=False, keep_last_n=2) as new:
            assert new.load_latest_state()[1]["step"] == 3
            new.save(model, step=4)
            assert [c.step for c in new.committed_checkpoints()] == [3, 4]
        assert names(tmp_path) == ["checkpoint_step00000003.pt", "checkpoint_step00000004.pt"]

    def test_unreadable_manifest_falls_back_to_the_directory_listing(self, tmp_path, caplog):
        model = SimpleModel()
        with manager(tmp_path, async_save=False) as mgr:
            mgr.save(model, step=1)
        (tmp_path / "manifest.json").write_text("{ not json")
        with manager(tmp_path, async_save=False) as mgr, caplog.at_level(logging.ERROR):
            assert mgr.load_latest_state()[1]["step"] == 1
        assert "falling back to the directory listing" in caplog.text


class TestLoadFailures:
    """Corrupted checkpoints are skipped. Unreachable ones are not, and 'none usable' is not 'none'."""

    def test_every_checkpoint_corrupted_is_an_error_not_a_fresh_start(self, tmp_path):
        model = SimpleModel()
        with manager(tmp_path, async_save=False) as mgr:
            corrupt(mgr.save(model, step=1))
            corrupt(mgr.save(model, step=2))
        with manager(tmp_path, async_save=False) as mgr:
            with pytest.raises(NoUsableCheckpointError, match="Refusing to silently start from step 0"):
                mgr.load_latest_state()
            with pytest.raises(NoUsableCheckpointError):
                mgr.try_load_latest(model)
        assert len(names(tmp_path)) == 2  # nothing was deleted

    def test_storage_errors_are_retried_then_raised_never_skipped(self, tmp_path, monkeypatch):
        import time

        monkeypatch.setattr(time, "sleep", lambda s: None)
        model = SimpleModel()
        with manager(tmp_path, async_save=False) as mgr:
            mgr.save(model, step=1)
            mgr.save(model, step=2)

        class Unreachable(LocalStorage):
            attempts = 0

            def load(self, path):
                Unreachable.attempts += 1
                raise OSError("stale NFS file handle")

        with manager(tmp_path, async_save=False, storage=Unreachable()) as mgr:
            with pytest.raises(OSError, match="stale NFS"):
                mgr.load_latest_state()
        assert Unreachable.attempts == 3  # the newest checkpoint, tried 1 + 2 times; older ones not consulted

    def test_transient_storage_error_recovers(self, tmp_path, monkeypatch):
        import time

        monkeypatch.setattr(time, "sleep", lambda s: None)
        model = SimpleModel()
        with manager(tmp_path, async_save=False) as mgr:
            mgr.save(model, step=7)

        class Flaky(LocalStorage):
            failures = 2

            def load(self, path):
                if Flaky.failures:
                    Flaky.failures -= 1
                    raise OSError("throttled")
                return super().load(path)

        with manager(tmp_path, async_save=False, storage=Flaky()) as mgr:
            assert mgr.load_latest_state()[1]["step"] == 7

    def test_untrusted_checkpoint_stops_the_resume_instead_of_falling_back(self, tmp_path):
        """Skipping it would quietly resume an older checkpoint and discard the newer work."""
        from flextrain.checkpoint import UntrustedCheckpointError

        model = SimpleModel()
        with manager(tmp_path, async_save=False) as mgr:
            mgr.save(model, step=1)
            mgr.save(model, step=2, extra={"hook": threading.Event})  # a class reference: needs real unpickling
        with manager(tmp_path, async_save=False) as mgr:
            with pytest.raises(UntrustedCheckpointError):
                mgr.load_latest_state()
        with manager(tmp_path, async_save=False, allow_pickle=True) as mgr:
            assert mgr.load_latest_state()[1]["step"] == 2

    def test_missing_file_named_by_the_manifest_is_skipped(self, tmp_path):
        model = SimpleModel()
        with manager(tmp_path, async_save=False) as mgr:
            mgr.save(model, step=1)
            Path(mgr.save(model, step=2)).unlink()
        with manager(tmp_path, async_save=False) as mgr:
            assert mgr.load_latest_state()[1]["step"] == 1


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
    (tmp_path / ".checkpoint_step00000007.pt.sha256.deadbeef.tmp").write_bytes(b"partial digest")
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
