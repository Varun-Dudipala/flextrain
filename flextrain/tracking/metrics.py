"""Append-only JSONL metrics logging."""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def _jsonable(value: Any) -> Any:
    """Convert tensors / numpy scalars to Python and non-finite floats to null (valid JSON)."""
    if hasattr(value, "item") and callable(value.item):
        try:
            value = value.item()
        except (ValueError, RuntimeError):
            return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class MetricsLogger:
    """Writes one JSON object per line to ``<run_dir>/metrics.jsonl``.

    JSONL is append-only and crash tolerant: a job killed mid-write loses at most the
    last line, and a resumed run simply keeps appending to the same file.
    """

    def __init__(self, run_dir: Union[str, Path], enabled: bool = True):
        self.run_dir = Path(run_dir)
        self.enabled = enabled
        self.metrics_file = self.run_dir / "metrics.jsonl"
        if enabled:
            self.run_dir.mkdir(parents=True, exist_ok=True)

    def log(self, metrics: Dict[str, Any], step: int, event: Optional[str] = None) -> None:
        if not self.enabled:
            return
        entry: Dict[str, Any] = {"step": int(step), "time": datetime.now(timezone.utc).isoformat()}
        if event:
            entry["event"] = event
        entry.update(_jsonable(metrics))
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        (self.run_dir / "hparams.json").write_text(json.dumps(_jsonable(params), indent=2))

    def read(self) -> List[Dict[str, Any]]:
        return read_metrics(self.metrics_file)


def read_metrics(path: Union[str, Path], tail: Optional[int] = None) -> List[Dict[str, Any]]:
    """Read a metrics JSONL file, skipping a torn last line from a crash."""
    path = Path(path)
    if not path.exists():
        return []
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                logger.debug("Skipping malformed metrics line in %s", path)
    return entries[-tail:] if tail else entries
