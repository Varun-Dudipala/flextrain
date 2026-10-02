"""Metrics logging and run registry."""

import json
import math

import pytest
import torch

from flextrain.tracking import MetricsLogger, RunRegistry, make_run_id, read_metrics


class TestMetricsLogger:
    def test_log_and_read(self, tmp_path):
        ml = MetricsLogger(tmp_path / "run")
        ml.log({"loss": 0.5, "acc": 0.8}, step=1)
        ml.log({"loss": 0.3}, step=2, event="eval")
        entries = ml.read()
        assert [e["step"] for e in entries] == [1, 2]
        assert entries[0]["loss"] == 0.5 and entries[1]["event"] == "eval"
        assert "time" in entries[0]

    def test_values_are_json_safe(self, tmp_path):
        ml = MetricsLogger(tmp_path)
        ml.log({"tensor": torch.tensor(1.5), "nan": float("nan"), "inf": math.inf, "nested": {"x": torch.tensor(2)}},
               step=0)
        line = (tmp_path / "metrics.jsonl").read_text()
        assert "NaN" not in line and "Infinity" not in line  # strict JSON for the dashboard
        entry = json.loads(line)
        assert entry["tensor"] == 1.5 and entry["nan"] is None and entry["nested"]["x"] == 2

    def test_append_across_instances(self, tmp_path):
        MetricsLogger(tmp_path).log({"loss": 1}, step=1)
        MetricsLogger(tmp_path).log({"loss": 2}, step=2)  # resumed run appends
        assert len(read_metrics(tmp_path / "metrics.jsonl")) == 2

    def test_torn_last_line_is_skipped(self, tmp_path):
        ml = MetricsLogger(tmp_path)
        ml.log({"loss": 1}, step=1)
        with open(ml.metrics_file, "a") as f:
            f.write('{"step": 2, "lo')  # process killed mid-write
        assert [e["step"] for e in ml.read()] == [1]

    def test_tail(self, tmp_path):
        ml = MetricsLogger(tmp_path)
        for i in range(10):
            ml.log({"loss": i}, step=i)
        assert [e["step"] for e in read_metrics(ml.metrics_file, tail=3)] == [7, 8, 9]

    def test_disabled_writes_nothing(self, tmp_path):
        ml = MetricsLogger(tmp_path / "x", enabled=False)
        ml.log({"loss": 1}, step=1)
        assert not (tmp_path / "x").exists() and ml.read() == []

    def test_hyperparameters(self, tmp_path):
        MetricsLogger(tmp_path).log_hyperparameters({"lr": 1e-3, "batch": 32})
        assert json.loads((tmp_path / "hparams.json").read_text())["lr"] == 1e-3

    def test_missing_file(self, tmp_path):
        assert read_metrics(tmp_path / "none.jsonl") == []


class TestRunRegistry:
    def test_update_merges_and_timestamps(self, tmp_path):
        reg = RunRegistry(tmp_path)
        reg.update("exp-run1", status="running", step=1)
        rec = reg.update("exp-run1", step=5, loss=0.1)
        assert rec["status"] == "running" and rec["step"] == 5 and rec["loss"] == 0.1
        assert rec["updated_at"] >= rec["created_at"]
        assert reg.get("exp-run1") == rec

    def test_list_sorted_by_recency_and_filtered(self, tmp_path):
        reg = RunRegistry(tmp_path)
        reg.update("a", status="completed")
        reg.update("b", status="running")
        assert [r["run_id"] for r in reg.list()] == ["b", "a"]
        assert [r["run_id"] for r in reg.list(status="completed")] == ["a"]

    def test_atomic_writes_leave_no_temp_files(self, tmp_path):
        reg = RunRegistry(tmp_path)
        for i in range(5):
            reg.update("r", step=i)
        assert [p.name for p in (tmp_path / "runs").iterdir()] == ["r.json"]

    def test_corrupt_record_is_ignored(self, tmp_path):
        reg = RunRegistry(tmp_path)
        reg.update("good", status="running")
        (tmp_path / "runs" / "bad.json").write_text("{not json")
        assert [r["run_id"] for r in reg.list()] == ["good"]
        assert reg.get("bad") is None

    @pytest.mark.parametrize("bad", ["../escape", "a/b", ""])
    def test_rejects_path_traversal(self, tmp_path, bad):
        with pytest.raises(ValueError):
            RunRegistry(tmp_path).get(bad)

    def test_delete(self, tmp_path):
        reg = RunRegistry(tmp_path)
        reg.update("r", status="running")
        assert reg.delete("r") and not reg.delete("r") and reg.list() == []

    def test_default_home_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FLEXTRAIN_HOME", str(tmp_path / "custom"))
        assert RunRegistry().home == tmp_path / "custom"

    def test_make_run_id_is_filesystem_safe(self):
        assert make_run_id("my exp", "2024/01/01 run") == "my-exp-2024-01-01-run"
