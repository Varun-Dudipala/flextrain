"""Elastic manager (batch geometry across resizes) and torchrun command construction."""

import sys

import pytest

from flextrain.config import Config, TrainingConfig
from flextrain.elastic import ElasticEnv, ElasticManager, build_torchrun_command


class TestElasticEnv:
    def test_reads_torchelastic_variables(self, monkeypatch):
        monkeypatch.setenv("TORCHELASTIC_RESTART_COUNT", "2")
        monkeypatch.setenv("TORCHELASTIC_MAX_RESTARTS", "3")
        monkeypatch.setenv("TORCHELASTIC_RUN_ID", "job")
        env = ElasticEnv.from_env()
        assert env.restart_count == 2 and env.max_restarts == 3 and env.is_restart and env.under_elastic_agent

    def test_defaults_outside_agent(self):
        env = ElasticEnv.from_env()
        assert env.restart_count == 0 and not env.is_restart and not env.under_elastic_agent


class TestElasticManager:
    def test_derives_accumulation_for_world_size(self):
        tc = TrainingConfig(batch_size=4, global_batch_size=32)
        assert ElasticManager(tc, 2, ElasticEnv()).accumulation_steps == 4
        assert ElasticManager(tc, 8, ElasticEnv()).accumulation_steps == 1

    def test_resize_preserves_global_batch(self):
        tc = TrainingConfig(batch_size=4, global_batch_size=32)
        change = ElasticManager(tc, 2, ElasticEnv()).on_resume({"world_size": 4, "global_batch": 32})
        assert change.resized and not change.global_batch_changed

    def test_resize_without_global_batch_warns(self, caplog):
        tc = TrainingConfig(batch_size=4, gradient_accumulation_steps=2)
        change = ElasticManager(tc, 2, ElasticEnv()).on_resume({"world_size": 4, "global_batch": 32})
        assert change.resized and change.global_batch_changed  # 4*2*2 = 16 != 32
        assert "Global batch changed" in caplog.text

    def test_indivisible_global_batch_warns(self, caplog):
        manager = ElasticManager(TrainingConfig(batch_size=4, global_batch_size=32), 3, ElasticEnv())
        assert manager.global_batch == 36 and "not divisible" in caplog.text


class TestTorchrunCommand:
    def test_single_node_is_standalone(self):
        cfg = Config()
        cfg.elastic.nproc_per_node = 4
        cmd = build_torchrun_command(cfg, "train.py", ["--lr", "1"])
        assert cmd[:3] == [sys.executable, "-m", "torch.distributed.run"]
        assert "--standalone" in cmd and "--nnodes=1" in cmd and "--nproc-per-node=4" in cmd
        assert cmd[-3:] == ["train.py", "--lr", "1"]

    def test_elastic_multi_node_uses_rendezvous(self):
        cfg = Config.from_dict({"main": {"experiment_name": "exp"},
                                "elastic": {"min_nodes": 2, "max_nodes": 4, "rdzv_endpoint": "head:29400",
                                            "max_restarts": 5}})
        cmd = build_torchrun_command(cfg, "train.py")
        assert "--nnodes=2:4" in cmd and "--rdzv-endpoint=head:29400" in cmd
        assert "--rdzv-id=exp" in cmd and "--rdzv-backend=c10d" in cmd and "--max-restarts=5" in cmd
        assert "--standalone" not in cmd

    def test_cli_overrides_win(self):
        cmd = build_torchrun_command(Config(), "t.py", nproc_per_node=2, max_restarts=0)
        assert "--nproc-per-node=2" in cmd and "--max-restarts=0" in cmd

    def test_multi_node_requires_endpoint(self):
        cfg = Config.from_dict({"elastic": {"min_nodes": 1, "max_nodes": 2}})
        with pytest.raises(ValueError, match="rdzv_endpoint"):
            build_torchrun_command(cfg, "train.py")


def test_restart_attempts_get_isolated_rendezvous_namespaces(monkeypatch):
    """A restarted worker must never read a peer address published by the previous attempt."""
    import socket
    from datetime import timedelta

    import torch.distributed as dist

    from flextrain.core.distributed import _elastic_attempt_store

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", str(port))
    monkeypatch.setenv("TORCHELASTIC_USE_AGENT_STORE", "False")  # rank 0 hosts the store
    monkeypatch.setenv("TORCHELASTIC_RESTART_COUNT", "0")
    first = _elastic_attempt_store(rank=0, world_size=1, timeout=timedelta(seconds=10))
    first.set("rank_1_addr", "stale")
    monkeypatch.setenv("TORCHELASTIC_RESTART_COUNT", "1")
    second = dist.PrefixStore("flextrain/attempt_1", dist.TCPStore("127.0.0.1", port, 1, is_master=False,
                                                                   timeout=timedelta(seconds=10)))
    assert second.check(["rank_1_addr"]) is False  # the stale key is invisible to the next attempt
    assert first.get("rank_1_addr") == b"stale"


def test_no_elastic_store_outside_agent():
    from datetime import timedelta

    from flextrain.core.distributed import _elastic_attempt_store

    assert _elastic_attempt_store(0, 1, timedelta(seconds=1)) is None
