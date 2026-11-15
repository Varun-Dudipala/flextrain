"""Elastic manager (batch geometry across resizes) and torchrun command construction."""

import sys

import pytest

from flextrain.config import Config, TrainingConfig
from flextrain.elastic import ElasticEnv, ElasticManager, build_torchrun_command, finalize_runs, launch
from flextrain.tracking import RunRegistry


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
        assert cmd[:3] == [sys.executable, "-m", "flextrain.elastic.agent"]  # torchrun + launcher watch
        assert "--standalone" in cmd and "--nnodes=1" in cmd and "--nproc-per-node=4" in cmd
        # Loopback rendezvous: the host's FQDN is not resolvable everywhere (e.g. macOS).
        assert "--local-addr=127.0.0.1" in cmd
        assert cmd[-3:] == ["train.py", "--lr", "1"]

    def test_elastic_multi_node_uses_rendezvous(self):
        cfg = Config.from_dict({"main": {"experiment_name": "exp"},
                                "elastic": {"min_nodes": 2, "max_nodes": 4, "rdzv_endpoint": "head:29400",
                                            "max_restarts": 5}})
        cmd = build_torchrun_command(cfg, "train.py")
        assert "--nnodes=2:4" in cmd and "--rdzv-endpoint=head:29400" in cmd
        assert "--rdzv-id=exp" in cmd and "--rdzv-backend=c10d" in cmd and "--max-restarts=5" in cmd
        assert "--standalone" not in cmd and not any(arg.startswith("--local-addr") for arg in cmd)

    def test_cli_overrides_win(self):
        cmd = build_torchrun_command(Config(), "t.py", nproc_per_node=2, max_restarts=0)
        assert "--nproc-per-node=2" in cmd and "--max-restarts=0" in cmd

    def test_multi_node_requires_endpoint(self):
        cfg = Config.from_dict({"elastic": {"min_nodes": 1, "max_nodes": 2}})
        with pytest.raises(ValueError, match="rdzv_endpoint"):
            build_torchrun_command(cfg, "train.py")


class TestLauncherSettlesRunStatus:
    """Rank 0 cannot record its own death, so the launcher gives dead runs a final status."""

    def test_running_run_of_this_launch_is_marked_failed(self, tmp_path):
        reg = RunRegistry(tmp_path)
        reg.update("exp-a", status="running", launch_id="L1", step=7)
        assert finalize_runs("L1", -9, reg) == ["exp-a"]
        record = reg.get("exp-a")
        assert record["status"] == "failed" and record["exit_code"] == -9 and record["finished_at"]
        assert "exited with code -9" in record["error"] and record["step"] == 7

    def test_final_statuses_and_their_errors_are_left_alone(self, tmp_path):
        reg = RunRegistry(tmp_path)
        reg.update("exp-failed", status="failed", launch_id="L1", error="RuntimeError: peer lost")
        reg.update("exp-done", status="completed", launch_id="L1")
        reg.update("exp-preempted", status="preempted", launch_id="L1")
        assert finalize_runs("L1", 1, reg) == []
        assert reg.get("exp-failed")["error"] == "RuntimeError: peer lost"
        assert reg.get("exp-done")["status"] == "completed" and reg.get("exp-preempted")["status"] == "preempted"

    def test_runs_of_other_launches_are_left_alone(self, tmp_path):
        reg = RunRegistry(tmp_path)
        reg.update("exp-other", status="running", launch_id="L2")
        reg.update("exp-unlaunched", status="running")  # started with plain `python train.py`
        assert finalize_runs("L1", 1, reg) == []
        assert reg.get("exp-other")["status"] == reg.get("exp-unlaunched")["status"] == "running"

    # A stand-in worker: registers a run under the launch id it was given, then dies or exits.
    WORKER = ("import os, sys\n"
              "from flextrain.tracking import RunRegistry\n"
              "RunRegistry().update('exp-w', status='running', launch_id=os.environ['FLEXTRAIN_LAUNCH_ID'])\n"
              "sys.exit(int(sys.argv[1]))\n")

    @pytest.mark.parametrize("exit_code,expected", [(3, "failed"), (0, "running")])
    def test_launch_exports_the_launch_id_and_settles_only_on_failure(self, exit_code, expected):
        assert launch([sys.executable, "-c", self.WORKER, str(exit_code)]) == exit_code
        record = RunRegistry().get("exp-w")
        assert record["status"] == expected and len(record["launch_id"]) == 32
        assert record.get("exit_code") == (exit_code if expected == "failed" else None)

    def test_launch_tells_the_agent_who_to_watch_and_to_use_fresh_stores(self):
        check = ("import os, sys\n"
                 "ok = (os.environ['FLEXTRAIN_LAUNCHER_PID'] == str(os.getppid())\n"
                 "      and os.environ['TORCH_DISABLE_SHARE_RDZV_TCP_STORE'] == '1'\n"
                 "      and os.environ['TORCH_ELASTIC_SHUTDOWN_TIMEOUT'] == '45')\n"
                 "sys.exit(0 if ok else 1)\n")
        assert launch([sys.executable, "-c", check], shutdown_timeout=45) == 0

    def test_launch_gives_its_workers_one_run_name(self):
        """Every incarnation of a launch shares it, including one restarted before any checkpoint."""
        check = ("import os, re, sys\n"
                 "sys.exit(0 if re.fullmatch(r'[0-9]{8}-[0-9]{6}', os.environ['FLEXTRAIN_RUN_NAME']) else 1)\n")
        assert launch([sys.executable, "-c", check]) == 0


# A stand-in for the agent. It registers a run, reports that it is ready, and on SIGTERM behaves
# as told: "preempt" records a clean stop and exits non-zero (as torchrun does after a signal),
# "die" exits non-zero without recording anything, "ignore" keeps running.
STAND_IN_AGENT = """
import os, signal, sys, time
from flextrain.tracking import RunRegistry
mode, ready = sys.argv[1], sys.argv[2]
registry = RunRegistry()
registry.update('exp-s', status='running', launch_id=os.environ['FLEXTRAIN_LAUNCH_ID'])
def on_term(signum, frame):
    if mode == 'preempt':
        registry.update('exp-s', status='preempted')
    if mode != 'ignore':
        sys.exit(1)
signal.signal(signal.SIGTERM, on_term)
open(ready, 'w').write(str(os.getpid()))
time.sleep(60)
"""


class TestLauncherStopSignals:
    """`flextrain launch` is what a scheduler or a terminal signals; what it returns is what they see."""

    def launch_and_signal(self, tmp_path, mode, times=1):
        import os
        import signal
        import threading
        import time

        ready = tmp_path / "ready"
        inside = threading.Event()

        def signal_once_ready():
            while not ready.exists() or not ready.read_text():
                time.sleep(0.05)
            for _ in range(times):
                if inside.is_set():  # never signal the test process after launch() has returned
                    os.kill(os.getpid(), signal.SIGTERM)
                    time.sleep(0.5)

        thread = threading.Thread(target=signal_once_ready, daemon=True)
        thread.start()
        inside.set()
        try:
            code = launch([sys.executable, "-c", STAND_IN_AGENT, mode, str(ready)])
        finally:
            inside.clear()
        thread.join(timeout=5)
        agent_pid = int(ready.read_text())
        return code, RunRegistry().get("exp-s"), agent_pid

    def test_clean_stop_on_request_exits_zero(self, tmp_path):
        """The agent exits non-zero after any signal; a job that stopped as asked and left a
        checkpoint is not a failure."""
        code, record, _ = self.launch_and_signal(tmp_path, "preempt")
        assert code == 0 and record["status"] == "preempted"

    def test_stop_that_did_not_end_cleanly_is_a_failure(self, tmp_path):
        code, record, _ = self.launch_and_signal(tmp_path, "die")
        assert code == 1 and record["status"] == "failed" and record["exit_code"] == 1

    def test_second_signal_stops_waiting_and_does_not_leave_the_run_running(self, tmp_path):
        import os
        import signal

        code, record, agent_pid = self.launch_and_signal(tmp_path, "ignore", times=2)
        try:
            assert code == 128 + signal.SIGTERM
            assert record["status"] == "failed" and "interrupted again" in record["error"]
        finally:
            os.kill(agent_pid, signal.SIGKILL)


class TestAgentWatchesItsLauncher:
    def test_watch_only_starts_for_the_parent_named_in_the_environment(self, monkeypatch):
        import os

        from flextrain.elastic.agent import watch_launcher_if_asked

        monkeypatch.delenv("FLEXTRAIN_LAUNCHER_PID", raising=False)
        assert watch_launcher_if_asked() is False  # run by hand / from a --dry-run command line
        monkeypatch.setenv("FLEXTRAIN_LAUNCHER_PID", str(os.getpid()))
        assert watch_launcher_if_asked() is False  # names some process that is not our parent

    def test_agent_stops_when_its_launcher_dies(self, tmp_path):
        """A middle process plays the launcher: it starts a watching child and is then killed.
        The child must notice and deliver SIGTERM to itself."""
        import subprocess
        import time

        child = ("import os, signal, sys, time\n"
                 "from flextrain.elastic.agent import watch_launcher_if_asked\n"
                 "signal.signal(signal.SIGTERM, lambda s, f: (open(sys.argv[1], 'w').write('stopped'), sys.exit(0)))\n"
                 "assert watch_launcher_if_asked()\n"
                 "time.sleep(60)\n")
        middle = ("import os, subprocess, sys, time\n"
                  "env = dict(os.environ, FLEXTRAIN_LAUNCHER_PID=str(os.getpid()))\n"
                  "subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2]], env=env,\n"
                  "                 start_new_session=True)\n"
                  "time.sleep(60)\n")
        marker = tmp_path / "stopped"
        launcher = subprocess.Popen([sys.executable, "-c", middle, child, str(marker)])
        time.sleep(3)  # let the child start watching
        launcher.kill()
        launcher.wait()
        deadline = time.time() + 15
        while time.time() < deadline and not marker.exists():
            time.sleep(0.1)
        assert marker.exists(), "the agent stand-in did not notice that its launcher was gone"


class TestRestartNamespace:
    """With one agent-hosted store for the whole job, a restarted worker must never read a peer
    address published by the previous attempt."""

    @pytest.fixture
    def agent_store(self, monkeypatch):
        from datetime import timedelta

        import torch.distributed as dist

        # Starting the store means connecting to it over loopback. On GitHub's macOS runner the
        # first such connection in the process has timed out while the ones after it were
        # immediate, so a slow start is retried rather than reported as a failure of the tests
        # below, which are about what is read from a store that is already up.
        for tries_left in (2, 1, 0):
            try:
                server = dist.TCPStore("127.0.0.1", 0, 1, is_master=True, timeout=timedelta(seconds=20),
                                       multi_tenant=True)
                break
            except dist.DistNetworkError:
                if not tries_left:
                    raise
        monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
        monkeypatch.setenv("MASTER_PORT", str(server.port))
        monkeypatch.setenv("TORCHELASTIC_USE_AGENT_STORE", "True")
        return server

    def attempt(self, monkeypatch, number):
        from datetime import timedelta

        from flextrain.core.distributed import _elastic_attempt_store

        monkeypatch.setenv("TORCHELASTIC_RESTART_COUNT", str(number))
        return _elastic_attempt_store(rank=1, world_size=2, timeout=timedelta(seconds=10))

    def test_attempts_do_not_see_each_others_keys(self, agent_store, monkeypatch):
        first = self.attempt(monkeypatch, 0)
        first.set("rank_1_addr", "stale")
        second = self.attempt(monkeypatch, 1)
        assert second.check(["rank_1_addr"]) is False
        second.set("rank_1_addr", "fresh")
        assert first.get("rank_1_addr") == b"stale" and second.get("rank_1_addr") == b"fresh"

    def test_control_the_same_attempt_does_see_them(self, agent_store, monkeypatch):
        """Without this the test above would also pass for stores that never share anything."""
        self.attempt(monkeypatch, 0).set("rank_1_addr", "published")
        assert self.attempt(monkeypatch, 0).get("rank_1_addr") == b"published"

    def test_no_namespace_when_each_attempt_gets_a_fresh_store(self, agent_store, monkeypatch):
        """What `flextrain launch` arranges: the store is not the agent's, so the per-node restart
        counter (which can differ between nodes) is never used as a namespace."""
        monkeypatch.setenv("TORCHELASTIC_USE_AGENT_STORE", "False")
        assert self.attempt(monkeypatch, 1) is None


def test_no_elastic_store_outside_agent():
    from datetime import timedelta

    from flextrain.core.distributed import _elastic_attempt_store

    assert _elastic_attempt_store(0, 1, timedelta(seconds=1)) is None
