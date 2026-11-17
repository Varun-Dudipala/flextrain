"""What ``flextrain launch`` runs: torchrun's elastic agent, plus a watch on the launcher.

``python -m flextrain.elastic.agent <args>`` is ``python -m torch.distributed.run <args>`` with
one addition. The launcher starts the agent in its own session so that it is the only process a
terminal's or scheduler's signal reaches, and forwards those signals itself. The cost of that
isolation is that killing the launcher (``kill -9``, a supervisor that kills the process group
it started) no longer reaches the agent, which would keep its workers running - and keep
restarting them - with nothing left to stop it.

So when ``$FLEXTRAIN_LAUNCHER_PID`` names this process's parent, a daemon thread watches for
that parent to go away and then sends this process ``SIGTERM``. The agent handles it the way it
handles any stop request: it passes the signal to the workers, which stop together at the next
step boundary and write a checkpoint. The variable is set only by ``flextrain launch``; run
by hand (or from a command printed by ``launch --dry-run``) the agent does not watch anything,
so it is safe under ``nohup`` and after its shell exits.
"""

from __future__ import annotations

import os
import signal
import sys
import threading
import time
from typing import List, Optional

from .launcher import LAUNCHER_PID_ENV_VAR

_POLL_SECONDS = 1.0


def _watch_launcher(launcher_pid: int) -> None:
    while os.getppid() == launcher_pid:
        time.sleep(_POLL_SECONDS)
    os.kill(os.getpid(), signal.SIGTERM)


def watch_launcher_if_asked() -> bool:
    """Start the watch if this process was started by the launcher named in the environment."""
    value = os.environ.get(LAUNCHER_PID_ENV_VAR, "")
    if not value.isdigit() or int(value) != os.getppid():
        return False
    threading.Thread(target=_watch_launcher, args=(int(value),), daemon=True,
                     name="flextrain-launcher-watch").start()
    return True


def main(argv: Optional[List[str]] = None) -> None:
    watch_launcher_if_asked()
    from torch.distributed.run import main as torchrun_main

    torchrun_main(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
