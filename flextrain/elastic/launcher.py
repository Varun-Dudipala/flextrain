"""Build and run ``torchrun`` (torch.distributed.run) commands from the elastic config."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Dict, List, Optional, Sequence

from flextrain.config import CONFIG_ENV_VAR, OVERRIDES_ENV_VAR, Config


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
    free port). Elastic jobs use ``--nnodes=MIN:MAX`` with a c10d rendezvous, so nodes may
    join or leave and the agent restarts the group with the new size.
    """
    ec = config.elastic
    nproc = nproc_per_node if nproc_per_node is not None else ec.nproc_per_node
    restarts = max_restarts if max_restarts is not None else ec.max_restarts
    endpoint = rdzv_endpoint if rdzv_endpoint is not None else ec.rdzv_endpoint
    nodes = nnodes if nnodes is not None else (
        str(ec.min_nodes) if ec.min_nodes == ec.max_nodes else f"{ec.min_nodes}:{ec.max_nodes}"
    )
    max_nodes = int(nodes.split(":")[-1])

    cmd = [sys.executable, "-m", "torch.distributed.run"]
    if max_nodes == 1 and endpoint is None:
        cmd.append("--standalone")
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


def launch(
    cmd: List[str],
    config_path: Optional[str] = None,
    overrides: Optional[Sequence[str]] = None,
    env: Optional[Dict[str, str]] = None,
) -> int:
    """Run a launcher command, exporting the config path and overrides for the training script.

    Workers pick both up via ``flextrain.load_config()`` with no arguments.
    """
    child_env = dict(os.environ if env is None else env)
    if config_path:
        child_env[CONFIG_ENV_VAR] = os.path.abspath(config_path)
    if overrides:
        child_env[OVERRIDES_ENV_VAR] = "\n".join(overrides)
    return subprocess.call(cmd, env=child_env)
