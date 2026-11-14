"""Build and run ``torchrun`` (torch.distributed.run) commands from the elastic config."""

from __future__ import annotations

import contextlib
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from typing import Dict, Iterator, List, Optional, Sequence

from flextrain.config import CONFIG_ENV_VAR, OVERRIDES_ENV_VAR, Config
from flextrain.tracking import (
    COMPLETED,
    FAILED,
    HOME_ENV_VAR,
    LAUNCH_ID_ENV_VAR,
    PREEMPTED,
    RUN_NAME_ENV_VAR,
    RUNNING,
    RunRegistry,
)

logger = logging.getLogger(__name__)

LOOPBACK_ADDR = "127.0.0.1"
# torchrun's elastic agent, wrapped so that it stops the job if the launcher goes away.
AGENT_MODULE = "flextrain.elastic.agent"
LAUNCHER_PID_ENV_VAR = "FLEXTRAIN_LAUNCHER_PID"  # tells the agent which parent to watch


def build_torchrun_command(
    config: Config,
    script: str,
    script_args: Sequence[str] = (),
    nproc_per_node: Optional[int] = None,
    nnodes: Optional[str] = None,
    max_restarts: Optional[int] = None,
    rdzv_endpoint: Optional[str] = None,
) -> List[str]:
    """Translate ``config.elastic`` (plus CLI overrides) into a torchrun argv.

    Single-node jobs without an explicit endpoint use ``--standalone`` (local rendezvous on a
    free port, reached over loopback). Elastic jobs use ``--nnodes=MIN:MAX`` with a c10d
    rendezvous, so nodes may join or leave and the agent restarts the group with the new size.
    """
    ec = config.elastic
    nproc = nproc_per_node if nproc_per_node is not None else ec.nproc_per_node
    restarts = max_restarts if max_restarts is not None else ec.max_restarts
    endpoint = rdzv_endpoint if rdzv_endpoint is not None else ec.rdzv_endpoint
    nodes = nnodes if nnodes is not None else (
        str(ec.min_nodes) if ec.min_nodes == ec.max_nodes else f"{ec.min_nodes}:{ec.max_nodes}"
    )
    max_nodes = int(nodes.split(":")[-1])

    cmd = [sys.executable, "-m", AGENT_MODULE]
    if max_nodes == 1 and endpoint is None:
        # Every worker runs on this host, so pin the rendezvous to loopback. Otherwise torchrun
        # advertises socket.getfqdn(), which on some machines (notably macOS, where it can be
        # an "...ip6.arpa" reverse name) does not resolve, and the workers hang at startup.
        cmd += ["--standalone", f"--local-addr={LOOPBACK_ADDR}"]
    else:
        if endpoint is None:
            raise ValueError("multi-node launch needs elastic.rdzv_endpoint (host:port of the rendezvous)")
        cmd += [
            f"--rdzv-backend={ec.rdzv_backend}",
            f"--rdzv-endpoint={endpoint}",
            f"--rdzv-id={ec.rdzv_id or config.main.experiment_name}",
        ]
    cmd += [
        f"--nnodes={nodes}",
        f"--nproc-per-node={nproc}",
        f"--max-restarts={restarts}",
        f"--monitor-interval={ec.monitor_interval:g}",
        script,
        *script_args,
    ]
    return cmd


def finalize_runs(launch_id: str, exit_code: int, registry: Optional[RunRegistry] = None,
                  reason: Optional[str] = None) -> List[str]:
    """Mark the runs of this launch that are still ``running`` as failed; return their ids.

    A run's status is written by rank 0 of the run. Rank 0 cannot record its own death: if it
    is killed (SIGKILL, the OOM killer, or the agent tearing the group down while rank 0 is
    blocked in a collective with a dead peer) the record would say ``running`` forever. The
    launcher outlives the workers, so it settles the status once the agent has given up.
    Runs that already recorded a final status (and their own, more specific error) are left alone.
    """
    registry = registry or RunRegistry()
    finalized = []
    for record in registry.list(status=RUNNING):
        if record.get("launch_id") != launch_id:
            continue
        registry.update(
            record["run_id"], status=FAILED, finished_at=time.time(), exit_code=exit_code,
            error=reason or f"worker group exited with code {exit_code} before the run recorded a final status",
        )
        finalized.append(record["run_id"])
    return finalized


class _StopWaiting(Exception):
    """A second stop signal arrived: stop waiting for the agent."""


@contextlib.contextmanager
def _forward_stop_signals(proc: subprocess.Popen) -> Iterator[List[int]]:
    """Make the launcher the single entry point for stop signals while the agent runs.

    The agent runs in its own session, so a terminal's Ctrl-C or a scheduler's SIGTERM reaches
    only the launcher, which forwards it exactly once. The agent passes it on to the workers,
    which stop together at the next step boundary and write one consistent checkpoint (see
    ``fault_tolerance.preemption``). Without this, SIGTERM to the launcher killed the launcher
    alone and left the agent and its workers running with nothing to stop them.

    SIGHUP (the terminal went away) is forwarded as SIGTERM, which the workers treat as a
    preemption request. A second signal stops the waiting, not the cleanup: the agent still
    terminates its workers within its own shutdown timeout.
    """
    received: List[int] = []  # yielded: the signals seen so far
    if threading.current_thread() is not threading.main_thread():
        yield received  # signal handlers can only be installed from the main thread
        return

    def handler(signum: int, frame: object) -> None:
        received.append(signum)
        if len(received) > 1:
            raise _StopWaiting(signum)
        # os.write, not logging: a handler must not take locks the interrupted code may hold.
        os.write(2, f"flextrain launch: {signal.Signals(signum).name} received, asking the job to checkpoint "
                    f"and stop (send it again to stop waiting)\n".encode())
        if os.name == "nt":
            proc.terminate()  # Windows has no signal to pass on
        else:
            proc.send_signal(signal.SIGINT if signum == signal.SIGINT else signal.SIGTERM)

    names = ("SIGINT", "SIGTERM", "SIGHUP")
    previous = {getattr(signal, n): signal.signal(getattr(signal, n), handler) for n in names if hasattr(signal, n)}
    try:
        yield received
    finally:
        for signum, old in previous.items():
            signal.signal(signum, old)


def _runs_of(launch_id: str, registry: RunRegistry) -> List[dict]:
    return [r for r in registry.list() if r.get("launch_id") == launch_id]


def launch(
    cmd: List[str],
    config_path: Optional[str] = None,
    overrides: Optional[Sequence[str]] = None,
    env: Optional[Dict[str, str]] = None,
    shutdown_timeout: Optional[int] = None,
) -> int:
    """Run the agent command and supervise it. Returns the exit code for ``flextrain launch``.

    * The config path and overrides are exported for the training script, which picks both up
      via ``flextrain.load_config()`` with no arguments.
    * Stop signals sent to the launcher are forwarded to the job (``_forward_stop_signals``),
      and the agent stops the job if the launcher itself disappears (``elastic.agent``).
    * A job that stopped *because it was asked to*, with every run recorded as ``preempted``
      (or already ``completed``), exits 0: it did what was requested and left a checkpoint. The
      agent's own exit code is non-zero in that case, as it is for any signal.
    * Otherwise a non-zero exit is passed through, and runs that never recorded a final status
      are marked failed (``finalize_runs``).
    """
    child_env = dict(os.environ if env is None else env)
    if config_path:
        child_env[CONFIG_ENV_VAR] = os.path.abspath(config_path)
    if overrides:
        child_env[OVERRIDES_ENV_VAR] = "\n".join(overrides)
    launch_id = uuid.uuid4().hex
    child_env[LAUNCH_ID_ENV_VAR] = launch_id
    child_env[LAUNCHER_PID_ENV_VAR] = str(os.getpid())
    # One name for every incarnation of this launch, including one restarted before any checkpoint.
    child_env.setdefault(RUN_NAME_ENV_VAR, time.strftime("%Y%m%d-%H%M%S"))
    # A fresh rendezvous store for every restart, instead of one store shared by all attempts
    # (see core.distributed._elastic_attempt_store for why).
    child_env.setdefault("TORCH_DISABLE_SHARE_RDZV_TCP_STORE", "1")
    if shutdown_timeout is not None:
        child_env.setdefault("TORCH_ELASTIC_SHUTDOWN_TIMEOUT", str(shutdown_timeout))
    home = child_env.get(HOME_ENV_VAR)
    registry = RunRegistry(home) if home else RunRegistry()

    proc = subprocess.Popen(cmd, env=child_env, start_new_session=True)
    signals: List[int] = []
    try:
        with _forward_stop_signals(proc) as signals:
            exit_code = proc.wait()
    except _StopWaiting as stop:
        # Asked twice: stop waiting. The agent still terminates its workers within its shutdown
        # timeout; if they do manage to record "preempted" afterwards, that overwrites this.
        exit_code = 128 + int(stop.args[0])
        _settle(launch_id, exit_code, registry, "the launcher was interrupted again while the job was stopping")
        return exit_code
    if exit_code == 0:
        return 0
    runs = _runs_of(launch_id, registry)
    if signals and runs and all(r.get("status") in (PREEMPTED, COMPLETED) for r in runs):
        return 0  # stopped on request, with a checkpoint
    _settle(launch_id, exit_code, registry)
    return exit_code


def _settle(launch_id: str, exit_code: int, registry: RunRegistry, reason: Optional[str] = None) -> None:
    try:
        for run_id in finalize_runs(launch_id, exit_code, registry, reason):
            logger.warning("Run %s marked failed: %s", run_id,
                           reason or f"its workers exited (code {exit_code}) without a final status")
    except Exception as e:  # bookkeeping must not mask the job's own exit code
        logger.warning("Could not finalize run status: %s", e)
