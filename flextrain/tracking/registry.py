"""File-based registry of training runs, shared by the trainer, CLI and dashboard.

Each run is one small JSON document at ``$FLEXTRAIN_HOME/runs/<run_id>.json``
(default ``~/.flextrain``), rewritten atomically by rank 0 of the run. No server or
database is required, and readers never observe a half-written record.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .metrics import _jsonable

HOME_ENV_VAR = "FLEXTRAIN_HOME"
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")

RUNNING, COMPLETED, PREEMPTED, FAILED = "running", "completed", "preempted", "failed"
STATUSES = (RUNNING, COMPLETED, PREEMPTED, FAILED)


def default_home() -> Path:
    return Path(os.environ.get(HOME_ENV_VAR) or Path.home() / ".flextrain")


def make_run_id(experiment: str, run_name: str) -> str:
    run_id = re.sub(r"[^A-Za-z0-9._-]+", "-", f"{experiment}-{run_name}").strip("-")
    return run_id or "run"


class RunRegistry:
    def __init__(self, home: Optional[Union[str, Path]] = None):
        self.home = Path(home) if home is not None else default_home()
        self.runs_dir = self.home / "runs"

    def _path(self, run_id: str) -> Path:
        if not _RUN_ID_RE.match(run_id):
            raise ValueError(f"invalid run id {run_id!r}")
        return self.runs_dir / f"{run_id}.json"

    def get(self, run_id: str) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(self._path(run_id).read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def update(self, run_id: str, **fields: Any) -> Dict[str, Any]:
        """Merge ``fields`` into the run record and write it atomically."""
        record = self.get(run_id) or {"run_id": run_id, "created_at": time.time()}
        record.update(_jsonable(fields))
        record["updated_at"] = time.time()
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.runs_dir, prefix=f".{run_id}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(record, f, indent=2)
            os.replace(tmp, self._path(run_id))
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return record

    def list(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self.runs_dir.is_dir():
            return []
        runs = []
        for path in self.runs_dir.glob("*.json"):
            try:
                record = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if status is None or record.get("status") == status:
                runs.append(record)
        return sorted(runs, key=lambda r: r.get("updated_at", 0), reverse=True)

    def delete(self, run_id: str) -> bool:
        path = self._path(run_id)
        if path.exists():
            path.unlink()
            return True
        return False
