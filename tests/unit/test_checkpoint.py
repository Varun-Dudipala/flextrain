"""Unit tests for checkpoint module."""

import pytest
import tempfile
import time
import torch
import torch.nn as nn
from pathlib import Path

from flextrain.config import CheckpointConfig
from flextrain.checkpoint import CheckpointManager
from flextrain.checkpoint.storage import LocalStorage, create_storage_backend


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, input_size=10, hidden_size=20, output_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class TestLocalStorage:
    """Tests for LocalStorage backend."""

    def test_save_and_load(self):
        storage = LocalStorage()
        state = {"key": "value", "tensor": torch.randn(10)}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            storage.save(state, path)

            assert path.exists()

            loaded = storage.load(path)
            assert loaded["key"] == "value"
            assert torch.equal(loaded["tensor"], state["tensor"])

    def test_exists(self):
        storage = LocalStorage()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            assert not storage.exists(path)

            storage.save({"key": "value"}, path)
            assert storage.exists(path)

    def test_delete(self):
        storage = LocalStorage()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            storage.save({"key": "value"}, path)
            assert path.exists()

            storage.delete(path)
            assert not path.exists()

    def test_load_nonexistent(self):
        storage = LocalStorage()
        with pytest.raises(FileNotFoundError):
            storage.load("/nonexistent/path.pt")


class TestCreateStorageBackend:
    """Tests for storage backend factory."""

    def test_create_local(self):
        backend = create_storage_backend("local")
        assert isinstance(backend, LocalStorage)

    def test_create_invalid(self):
        with pytest.raises(ValueError):
            create_storage_backend("invalid")


class TestCheckpointManager:
    """Tests for CheckpointManager."""

    def test_save_and_load(self):
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Get initial weights
        initial_weights = model.fc1.weight.clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Use sync save for predictable testing
            config = CheckpointConfig(checkpoint_dir=tmpdir, async_save=False)
            manager = CheckpointManager(config, rank=0, world_size=1)

            # Save checkpoint
            path = manager.save(
                model, optimizer,
                step=100,
                epoch=5,
                metrics={"loss": 0.5}
            )
            assert path is not None
            assert Path(path).exists()

            # Modify model weights
            with torch.no_grad():
                model.fc1.weight.fill_(0)

            # Load checkpoint
            result = manager.load(model, optimizer)

            assert result["step"] == 100
            assert result["epoch"] == 5
            assert result["metrics"]["loss"] == 0.5

            # Verify weights restored
            assert torch.equal(model.fc1.weight, initial_weights)

    def test_get_latest(self):
        model = SimpleModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(checkpoint_dir=tmpdir, async_save=False)
            manager = CheckpointManager(config, rank=0, world_size=1)

            # Save multiple checkpoints with small delays to ensure different timestamps
            manager.save(model, step=100)
            time.sleep(0.1)
            manager.save(model, step=200)
            time.sleep(0.1)
            manager.save(model, step=300)

            # Latest should be step 300
            latest = manager._get_latest()
            assert latest is not None
            assert "step00000300" in latest

    def test_pruning(self):
        model = SimpleModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(checkpoint_dir=tmpdir, keep_last_n=2, async_save=False)
            manager = CheckpointManager(config, rank=0, world_size=1)

            # Save more than keep_last_n checkpoints with small delays
            manager.save(model, step=100)
            time.sleep(0.1)
            manager.save(model, step=200)
            time.sleep(0.1)
            manager.save(model, step=300)

            # Should only keep 2
            checkpoints = list(Path(tmpdir).glob("checkpoint_*.pt"))
            assert len(checkpoints) == 2

    def test_try_load_latest_empty(self):
        model = SimpleModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(checkpoint_dir=tmpdir, async_save=False)
            manager = CheckpointManager(config, rank=0, world_size=1)

            result = manager.try_load_latest(model)
            assert result is None

    def test_save_with_scheduler(self):
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

        # Step scheduler a few times
        for _ in range(5):
            scheduler.step()

        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(checkpoint_dir=tmpdir, async_save=False)
            manager = CheckpointManager(config, rank=0, world_size=1)

            manager.save(model, optimizer, scheduler, step=100)

            # Create new scheduler
            new_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

            result = manager.load(model, optimizer, new_scheduler)
            assert result["step"] == 100


class TestAsyncCheckpointing:
    """Tests for async checkpointing functionality."""

    def test_async_save_and_wait(self):
        model = SimpleModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(checkpoint_dir=tmpdir, async_save=True, num_io_workers=1)
            manager = CheckpointManager(config, rank=0, world_size=1)

            # Save with async (non-blocking)
            path = manager.save(model, step=100, blocking=False)

            # Wait for async completion
            manager.wait_for_pending()

            # Now file should exist
            assert Path(path).exists()
            assert manager.pending_saves() == 0

    def test_pending_count(self):
        model = SimpleModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(checkpoint_dir=tmpdir, async_save=True, num_io_workers=1)
            manager = CheckpointManager(config, rank=0, world_size=1)

            # Initially no pending saves
            assert manager.pending_saves() == 0

            # After async save, should have pending
            manager.save(model, step=100, blocking=False)
            # Note: this might be 0 or 1 depending on timing, just ensure it works
            manager.wait_for_pending()
            assert manager.pending_saves() == 0

    def test_blocking_override(self):
        model = SimpleModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(checkpoint_dir=tmpdir, async_save=True)
            manager = CheckpointManager(config, rank=0, world_size=1)

            # Force blocking save even with async enabled
            path = manager.save(model, step=100, blocking=True)

            # File should exist immediately
            assert Path(path).exists()
