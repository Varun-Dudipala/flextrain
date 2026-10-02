"""Background checkpoint writer: non-blocking, ordered, bounded, failures surface, drains on close."""

import threading
import time

import pytest
import torch

from flextrain.checkpoint import AsyncCheckpointWriter, CheckpointSaveError, LocalStorage


class GatedStorage(LocalStorage):
    """Blocks every write until the test opens the gate; records write order."""

    def __init__(self):
        super().__init__(fsync=False)
        self.gate = threading.Event()
        self.order = []

    def save(self, state, path):
        assert self.gate.wait(timeout=10), "gate never opened"
        self.order.append(state["step"])
        return super().save(state, path)


class FlakyStorage(LocalStorage):
    def __init__(self, failures: int):
        super().__init__(fsync=False)
        self.failures = failures
        self.attempts = 0

    def save(self, state, path):
        self.attempts += 1
        if self.attempts <= self.failures:
            raise OSError("transient")
        return super().save(state, path)


def test_submit_returns_before_write_completes(tmp_path):
    storage = GatedStorage()
    writer = AsyncCheckpointWriter(storage, max_pending=1)
    start = time.perf_counter()
    future = writer.submit({"step": 1, "w": torch.ones(10)}, str(tmp_path / "a.pt"))
    assert time.perf_counter() - start < 1.0
    assert not future.done() and writer.pending_count() == 1
    assert not (tmp_path / "a.pt").exists()
    storage.gate.set()
    writer.wait()
    assert (tmp_path / "a.pt").exists() and writer.pending_count() == 0
    writer.close()


def test_saved_state_is_a_snapshot_not_a_reference(tmp_path):
    storage = GatedStorage()
    writer = AsyncCheckpointWriter(storage)
    live = {"step": 1, "w": torch.zeros(4)}
    writer.submit(live, str(tmp_path / "a.pt"))
    live["w"].add_(42)  # training keeps mutating parameters while the write is pending
    storage.gate.set()
    writer.close()
    assert torch.equal(torch.load(tmp_path / "a.pt")["w"], torch.zeros(4))


def test_writes_complete_in_submission_order(tmp_path):
    storage = GatedStorage()
    writer = AsyncCheckpointWriter(storage, max_pending=4)
    for step in range(4):
        writer.submit({"step": step}, str(tmp_path / f"{step}.pt"))
    storage.gate.set()
    writer.close()
    assert storage.order == [0, 1, 2, 3]


def test_backpressure_blocks_when_max_pending_reached(tmp_path):
    storage = GatedStorage()
    writer = AsyncCheckpointWriter(storage, max_pending=1)
    writer.submit({"step": 0}, str(tmp_path / "0.pt"))
    released = threading.Event()

    def second_submit():
        writer.submit({"step": 1}, str(tmp_path / "1.pt"))
        released.set()

    t = threading.Thread(target=second_submit)
    t.start()
    assert not released.wait(0.3), "second submit must block while the first write is in flight"
    storage.gate.set()
    assert released.wait(5)
    t.join()
    writer.close()
    assert storage.order == [0, 1]


def test_close_drains_every_queued_write(tmp_path):
    storage = GatedStorage()
    writer = AsyncCheckpointWriter(storage, max_pending=5)
    for step in range(5):
        writer.submit({"step": step}, str(tmp_path / f"{step}.pt"))
    threading.Timer(0.2, storage.gate.set).start()
    writer.close()
    assert sorted(p.name for p in tmp_path.glob("*.pt")) == [f"{s}.pt" for s in range(5)]


def test_failure_surfaces_on_wait_then_clears(tmp_path):
    writer = AsyncCheckpointWriter(FlakyStorage(failures=10), write_retries=0)
    writer.submit({"step": 1}, str(tmp_path / "a.pt"))
    with pytest.raises(CheckpointSaveError, match="transient"):
        writer.wait()
    writer.wait()  # reported once
    writer.close()


def test_failure_surfaces_on_next_submit(tmp_path):
    writer = AsyncCheckpointWriter(FlakyStorage(failures=10), write_retries=0)
    writer.submit({"step": 1}, str(tmp_path / "a.pt"))
    while writer.pending_count():
        time.sleep(0.01)
    with pytest.raises(CheckpointSaveError):
        writer.submit({"step": 2}, str(tmp_path / "b.pt"))
    writer.close(raise_errors=False)


def test_transient_failures_are_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    storage = FlakyStorage(failures=2)
    writer = AsyncCheckpointWriter(storage, write_retries=2)
    writer.submit({"step": 1}, str(tmp_path / "a.pt"))
    writer.close()
    assert storage.attempts == 3 and (tmp_path / "a.pt").exists()


def test_on_success_hook_runs_after_durable_write_and_its_errors_do_not_fail_the_save(tmp_path):
    seen = []

    def hook(path):
        seen.append((path, (tmp_path / "a.pt").exists()))
        raise RuntimeError("pruning broke")

    writer = AsyncCheckpointWriter(LocalStorage(fsync=False))
    writer.submit({"step": 1}, str(tmp_path / "a.pt"), on_success=hook)
    writer.close()  # does not raise
    assert seen == [(str(tmp_path / "a.pt"), True)]


def test_history_records_timings(tmp_path):
    writer = AsyncCheckpointWriter(LocalStorage(fsync=False))
    writer.submit({"step": 1, "w": torch.ones(1000)}, str(tmp_path / "a.pt"))
    writer.close()
    record = writer.history[-1]
    assert record.nbytes > 4000 and record.write_seconds > 0 and record.snapshot_seconds >= 0


def test_submit_after_close_raises(tmp_path):
    writer = AsyncCheckpointWriter(LocalStorage())
    writer.close()
    with pytest.raises(RuntimeError):
        writer.submit({"step": 1}, str(tmp_path / "a.pt"))


def test_invalid_max_pending():
    with pytest.raises(ValueError):
        AsyncCheckpointWriter(LocalStorage(), max_pending=0)
