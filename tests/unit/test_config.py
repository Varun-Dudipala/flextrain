"""Unit tests for configuration module."""

import pytest
import tempfile
from pathlib import Path

from flextrain.config import (
    Config,
    TrainingConfig,
    DistributedConfig,
    CheckpointConfig,
    ElasticConfig,
    load_config,
)


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_values(self):
        config = TrainingConfig()
        assert config.batch_size == 32
        assert config.learning_rate == 1e-4
        assert config.optimizer == "adamw"

    def test_custom_values(self):
        config = TrainingConfig(batch_size=64, learning_rate=3e-4)
        assert config.batch_size == 64
        assert config.learning_rate == 3e-4

    def test_validation_batch_size(self):
        with pytest.raises(ValueError):
            TrainingConfig(batch_size=0)

    def test_validation_learning_rate(self):
        with pytest.raises(ValueError):
            TrainingConfig(learning_rate=-1)

    def test_effective_batch_size(self):
        config = TrainingConfig(
            batch_size=32,
            micro_batch_size=8,
            gradient_accumulation_steps=4
        )
        assert config.effective_batch_size == 32  # 8 * 4


class TestDistributedConfig:
    """Tests for DistributedConfig."""

    def test_default_strategy(self):
        config = DistributedConfig()
        assert config.strategy == "ddp"

    def test_fsdp_strategy(self):
        config = DistributedConfig(strategy="fsdp")
        assert config.strategy == "fsdp"

    def test_invalid_strategy(self):
        with pytest.raises(ValueError):
            DistributedConfig(strategy="invalid")

    def test_mixed_precision(self):
        config = DistributedConfig(mixed_precision=True, mixed_precision_dtype="bf16")
        assert config.mixed_precision is True
        assert config.mixed_precision_dtype == "bf16"


class TestCheckpointConfig:
    """Tests for CheckpointConfig."""

    def test_default_values(self):
        config = CheckpointConfig()
        assert config.storage_backend == "local"
        assert config.async_save is True
        assert config.keep_last_n == 5

    def test_invalid_backend(self):
        with pytest.raises(ValueError):
            CheckpointConfig(storage_backend="invalid")

    def test_keep_last_n_validation(self):
        with pytest.raises(ValueError):
            CheckpointConfig(keep_last_n=0)


class TestElasticConfig:
    """Tests for ElasticConfig."""

    def test_default_values(self):
        config = ElasticConfig()
        assert config.min_nodes == 1
        assert config.max_nodes == 1

    def test_invalid_min_nodes(self):
        with pytest.raises(ValueError):
            ElasticConfig(min_nodes=0)

    def test_invalid_max_nodes(self):
        with pytest.raises(ValueError):
            ElasticConfig(min_nodes=4, max_nodes=2)


class TestConfig:
    """Tests for main Config class."""

    def test_default_config(self):
        config = Config()
        assert config.training.batch_size == 32
        assert config.distributed.strategy == "ddp"

    def test_to_dict(self):
        config = Config()
        d = config.to_dict()
        assert "training" in d
        assert "distributed" in d
        assert "checkpoint" in d

    def test_to_yaml(self):
        config = Config()
        yaml_str = config.to_yaml()
        assert "training:" in yaml_str
        assert "batch_size:" in yaml_str

    def test_to_yaml_file(self):
        config = Config()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            config.to_yaml(path)
            assert path.exists()

    def test_load_config(self):
        config = Config()
        config.training.batch_size = 64

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            config.to_yaml(path)

            loaded = load_config(path)
            assert loaded.training.batch_size == 64
