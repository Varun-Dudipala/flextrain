"""Unit tests for tracking module."""

import pytest
import tempfile
import json
from pathlib import Path

from flextrain.tracking import MetricsLogger


class TestMetricsLogger:
    """Tests for MetricsLogger."""

    def test_initialization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(
                experiment_name="test_exp",
                run_name="run_001",
                log_dir=tmpdir
            )
            assert logger.experiment_name == "test_exp"
            assert logger.run_name == "run_001"

    def test_log_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(log_dir=tmpdir)

            logger.log({"loss": 0.5, "accuracy": 0.8}, step=1)
            logger.log({"loss": 0.3, "accuracy": 0.9}, step=2)

            metrics = logger.get_metrics()
            assert len(metrics) == 2
            assert metrics[0]["loss"] == 0.5
            assert metrics[1]["loss"] == 0.3

    def test_log_without_step(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(log_dir=tmpdir)

            logger.log({"loss": 0.5})
            logger.log({"loss": 0.4})

            metrics = logger.get_metrics()
            assert metrics[0]["step"] == 0
            assert metrics[1]["step"] == 1

    def test_log_hyperparameters(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(
                experiment_name="test",
                run_name="run1",
                log_dir=tmpdir
            )

            params = {
                "learning_rate": 0.001,
                "batch_size": 32,
                "model": "gpt2"
            }
            logger.log_hyperparameters(params)

            params_file = Path(tmpdir) / "test_run1_params.json"
            assert params_file.exists()

            with open(params_file) as f:
                loaded = json.load(f)
            assert loaded["learning_rate"] == 0.001

    def test_metrics_file_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(log_dir=tmpdir)
            logger.log({"loss": 0.5}, step=10)

            metrics = logger.get_metrics()
            assert "timestamp" in metrics[0]
            assert "step" in metrics[0]
            assert metrics[0]["step"] == 10

    def test_empty_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(log_dir=tmpdir)
            metrics = logger.get_metrics()
            assert metrics == []
