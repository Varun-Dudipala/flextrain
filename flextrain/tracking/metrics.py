"""Metrics logging and experiment tracking."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import json

logger = logging.getLogger(__name__)


class MetricsLogger:
    """Simple metrics logger with SQLite backend."""

    def __init__(
        self,
        experiment_name: str = "default",
        run_name: Optional[str] = None,
        log_dir: str = "./logs",
    ):
        self.experiment_name = experiment_name
        self.run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._metrics_file = self.log_dir / f"{experiment_name}_{self.run_name}.jsonl"
        self._step = 0

        logger.info(f"MetricsLogger initialized: {self._metrics_file}")

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics."""
        step = step if step is not None else self._step
        self._step = step + 1

        entry = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            **metrics,
        }

        with open(self._metrics_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters."""
        params_file = self.log_dir / f"{self.experiment_name}_{self.run_name}_params.json"
        with open(params_file, "w") as f:
            json.dump(params, f, indent=2, default=str)

    def get_metrics(self) -> list:
        """Get all logged metrics."""
        if not self._metrics_file.exists():
            return []

        metrics = []
        with open(self._metrics_file, "r") as f:
            for line in f:
                metrics.append(json.loads(line))
        return metrics
