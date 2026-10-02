"""FastAPI service exposing the run registry and per-run metrics, plus a small dashboard."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

from flextrain import __version__
from flextrain.tracking import RUNNING, RunRegistry, read_metrics

from .dashboard import DASHBOARD_HTML

# A running job whose record has not been touched for this long is reported as stale
# (it probably died without being able to mark itself failed, e.g. SIGKILL / OOM-killer).
STALE_AFTER_SECONDS = 600


def _pid_alive(pid: Any) -> Optional[bool]:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except (PermissionError, TypeError, ValueError, OSError):
        return None
    return True


def _with_liveness(record: Dict[str, Any], hostname: str) -> Dict[str, Any]:
    record = dict(record)
    age = time.time() - float(record.get("updated_at", 0))
    record["seconds_since_update"] = round(age, 1)
    stale = False
    if record.get("status") == RUNNING:
        if record.get("host") == hostname and _pid_alive(record.get("pid")) is False:
            stale = True
        elif age > STALE_AFTER_SECONDS:
            stale = True
    record["stale"] = stale
    return record


def create_app(registry: Optional[RunRegistry] = None) -> FastAPI:
    registry = registry or RunRegistry()
    hostname = os.uname().nodename if hasattr(os, "uname") else ""
    app = FastAPI(title="FlexTrain", version=__version__)

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> str:
        return DASHBOARD_HTML

    @app.get("/api/health")
    def health() -> Dict[str, Any]:
        return {"status": "healthy", "version": __version__}

    @app.get("/api/jobs")
    def list_jobs(status: Optional[str] = Query(None, description="filter by status")) -> Dict[str, Any]:
        jobs = [_with_liveness(r, hostname) for r in registry.list()]
        counts: Dict[str, int] = {}
        for job in jobs:
            counts[job.get("status", "unknown")] = counts.get(job.get("status", "unknown"), 0) + 1
        if status is not None:
            jobs = [j for j in jobs if j.get("status") == status]
        return {"jobs": jobs, "total": len(jobs), "counts": counts}

    @app.get("/api/jobs/{run_id}")
    def get_job(run_id: str) -> Dict[str, Any]:
        try:
            record = registry.get(run_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="invalid run id") from None
        if record is None:
            raise HTTPException(status_code=404, detail=f"run {run_id!r} not found")
        return _with_liveness(record, hostname)

    @app.get("/api/jobs/{run_id}/metrics")
    def get_metrics(run_id: str, tail: Optional[int] = Query(None, ge=1, le=100_000)) -> Dict[str, List]:
        record = get_job(run_id)
        path = record.get("metrics_path")
        entries = read_metrics(Path(path), tail=tail) if path else []
        return {"metrics": entries}

    return app


app = create_app()
