"""REST API tests (FastAPI TestClient against a temporary registry)."""

import os
import time

import pytest
from fastapi.testclient import TestClient

from flextrain import __version__
from flextrain.api.server import STALE_AFTER_SECONDS, create_app
from flextrain.tracking import MetricsLogger, RunRegistry


@pytest.fixture
def registry(tmp_path):
    return RunRegistry(tmp_path / "home")


@pytest.fixture
def client(registry):
    return TestClient(create_app(registry))


def test_health(client):
    assert client.get("/api/health").json() == {"status": "healthy", "version": __version__}


def test_dashboard_served(client):
    res = client.get("/")
    assert res.status_code == 200 and "FlexTrain" in res.text and "/api/jobs" in res.text


def test_jobs_listing_counts_and_filter(client, registry):
    registry.update("exp-a", status="running", host=os.uname().nodename, pid=os.getpid(), step=3)
    registry.update("exp-b", status="completed", step=10)
    body = client.get("/api/jobs").json()
    assert body["total"] == 2 and body["counts"] == {"running": 1, "completed": 1}
    running = client.get("/api/jobs", params={"status": "running"}).json()
    assert [j["run_id"] for j in running["jobs"]] == ["exp-a"]
    assert running["jobs"][0]["stale"] is False


def test_dead_process_is_reported_stale(client, registry):
    registry.update("exp-dead", status="running", host=os.uname().nodename, pid=2**22 + 12345)
    assert client.get("/api/jobs/exp-dead").json()["stale"] is True


def test_silent_job_is_reported_stale(client, registry):
    registry.update("exp-old", status="running", host="other-host", pid=1)
    record = registry.get("exp-old")
    record["updated_at"] = time.time() - STALE_AFTER_SECONDS - 1
    (registry.runs_dir / "exp-old.json").write_text(__import__("json").dumps(record))
    assert client.get("/api/jobs/exp-old").json()["stale"] is True


def test_get_job_404_and_400(client):
    assert client.get("/api/jobs/missing").status_code == 404
    assert client.get("/api/jobs/bad..name!").status_code == 400


def test_metrics_endpoint(client, registry, tmp_path):
    logger = MetricsLogger(tmp_path / "run")
    for step in range(5):
        logger.log({"loss": 1.0 / (step + 1)}, step=step)
    registry.update("exp-m", status="running", metrics_path=str(logger.metrics_file))
    assert len(client.get("/api/jobs/exp-m/metrics").json()["metrics"]) == 5
    tail = client.get("/api/jobs/exp-m/metrics", params={"tail": 2}).json()["metrics"]
    assert [m["step"] for m in tail] == [3, 4]


def test_metrics_for_run_without_file(client, registry):
    registry.update("exp-n", status="running")
    assert client.get("/api/jobs/exp-n/metrics").json() == {"metrics": []}
