"""Preemption handling, divergence guard and single-process collective helpers."""

import os
import signal
import threading

import pytest

from flextrain.core.distributed import all_gather_object, all_reduce_tensor, broadcast_object, init_distributed
from flextrain.fault_tolerance import NonFiniteGuard, PreemptionHandler, TrainingDivergedError


class TestPreemptionHandler:
    def test_signal_sets_flag_without_exiting(self):
        handler = PreemptionHandler(["SIGUSR1"])
        with handler:
            assert not handler.stop_requested
            os.kill(os.getpid(), signal.SIGUSR1)
            assert handler.stop_requested and handler.received_signal == "SIGUSR1"
            assert handler.should_stop()  # single process: the local flag is the global decision

    def test_previous_handlers_restored(self):
        sentinel = []
        previous = signal.signal(signal.SIGUSR1, lambda *a: sentinel.append(1))
        try:
            with PreemptionHandler(["SIGUSR1"]):
                assert signal.getsignal(signal.SIGUSR1) != previous
            os.kill(os.getpid(), signal.SIGUSR1)
            assert sentinel == [1]
        finally:
            signal.signal(signal.SIGUSR1, previous)

    def test_second_sigint_forces_keyboard_interrupt(self):
        handler = PreemptionHandler(["SIGINT"])
        with handler:
            os.kill(os.getpid(), signal.SIGINT)
            assert handler.stop_requested
            with pytest.raises(KeyboardInterrupt):
                os.kill(os.getpid(), signal.SIGINT)

    def test_install_off_main_thread_is_a_noop(self):
        handler = PreemptionHandler(["SIGUSR1"])
        result = []
        t = threading.Thread(target=lambda: result.append(handler.install()))
        t.start()
        t.join()
        assert result == [False]

    def test_programmatic_request(self):
        handler = PreemptionHandler([])
        handler.request_stop()
        assert handler.should_stop() and handler.received_signal == "SIGTERM"


# Run in a child process: the watchdog's job is to exit the process it lives in.
WATCHDOG_SCRIPT = """
import os, signal, sys, time
from flextrain.fault_tolerance import PreemptionHandler
mode = sys.argv[1]
handler = PreemptionHandler(['SIGTERM'], grace_seconds=1.0)
handler.install()
os.kill(os.getpid(), signal.SIGTERM)
deadline = time.monotonic() + 3.0
if mode == 'agreed':
    handler.stop_agreed()
if mode == 'blocked':
    # Like a rank whose peer died: the main thread sits inside a C++ call that does not return
    # to the interpreter, so the Python-level signal handler cannot run at all.
    from datetime import timedelta
    import torch.distributed as dist
    store = dist.TCPStore('127.0.0.1', 0, 1, is_master=True, timeout=timedelta(seconds=20))
    try:
        store.get('never-set')
    except Exception:
        pass
    print('unblocked after the full timeout')
while time.monotonic() < deadline:
    if mode == 'progress':
        handler.heartbeat()
    if mode == 'busy':
        with handler.busy():
            time.sleep(3.0)
    time.sleep(0.05)
print('survived')
"""


class TestStopWatchdog:
    """A rank blocked in a C++ collective cannot run a Python signal handler. With the handler
    installed it would ignore the agent's SIGTERM for the whole shutdown timeout."""

    def run_mode(self, mode):
        import subprocess
        import sys

        return subprocess.run([sys.executable, "-c", WATCHDOG_SCRIPT, mode], capture_output=True, text=True, timeout=60)

    @pytest.mark.parametrize("mode", ["stuck", "blocked"])
    def test_a_rank_that_makes_no_progress_after_the_signal_exits(self, mode):
        """'blocked' is the case that matters: the main thread is inside C++ for 20 s and cannot
        run the Python handler, so the watchdog has to learn of the signal some other way."""
        import time

        started = time.monotonic()
        proc = self.run_mode(mode)
        assert proc.returncode == 128 + signal.SIGTERM, proc.stderr
        assert "survived" not in proc.stdout and "unblocked" not in proc.stdout
        assert "No progress" in proc.stderr and "Received SIGTERM" in proc.stderr
        assert time.monotonic() - started < 15, "exited only when the blocking call timed out"

    @pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
    def test_forked_children_do_not_inherit_the_handler_or_the_wakeup_pipe(self):
        """A DataLoader worker is not a rank: SIGTERM must kill it as usual, and a signal sent to it
        must not show up in the parent's watchdog as the parent's own."""
        with PreemptionHandler(["SIGTERM"], grace_seconds=30):
            pid = os.fork()
            if pid == 0:  # child
                ok = signal.getsignal(signal.SIGTERM) == signal.SIG_DFL and signal.set_wakeup_fd(-1) == -1
                os._exit(0 if ok else 1)
            assert os.waitpid(pid, 0)[1] == 0
            assert signal.getsignal(signal.SIGTERM) != signal.SIG_DFL  # the parent keeps its handler

    @pytest.mark.parametrize("mode", ["progress", "agreed", "busy"])
    def test_a_rank_that_is_working_is_left_alone(self, mode):
        """Still stepping towards the stop, already agreed and saving, or inside a long save."""
        proc = self.run_mode(mode)
        assert proc.returncode == 0 and "survived" in proc.stdout, proc.stderr

    def test_disabled_without_a_grace_period(self):
        handler = PreemptionHandler(["SIGUSR1"])  # grace_seconds=None
        with handler:
            os.kill(os.getpid(), signal.SIGUSR1)
            import time

            time.sleep(0.6)  # several watchdog polls
            assert handler.stop_requested


class TestNonFiniteGuard:
    def test_finite_steps_pass(self):
        guard = NonFiniteGuard(max_consecutive=2)
        assert not guard.should_skip(1.0, step=1)

    def test_skips_then_raises_after_limit(self):
        guard = NonFiniteGuard(max_consecutive=2)
        assert guard.should_skip(float("nan"), 1)
        assert guard.should_skip(float("inf"), 2)
        with pytest.raises(TrainingDivergedError):
            guard.should_skip(float("nan"), 3)

    def test_finite_step_resets_streak(self):
        guard = NonFiniteGuard(max_consecutive=1)
        guard.should_skip(float("nan"), 1)
        guard.should_skip(0.5, 2)
        assert guard.should_skip(float("nan"), 3)  # streak restarted, no raise
        assert guard.total_skipped == 2

    def test_disabled(self):
        assert not NonFiniteGuard(enabled=False).should_skip(float("nan"), 1)

    def test_state_round_trip(self):
        guard = NonFiniteGuard()
        guard.should_skip(float("nan"), 1)
        other = NonFiniteGuard()
        other.load_state_dict(guard.state_dict())
        assert other.consecutive == 1 and other.total_skipped == 1


class TestSingleProcessCollectives:
    def test_helpers_are_identity_without_process_group(self):
        assert all_reduce_tensor([1.0, 2.0]) == [1.0, 2.0]
        assert broadcast_object({"a": 1}) == {"a": 1}
        assert all_gather_object(3) == [3]

    def test_init_distributed_single_process(self):
        ctx = init_distributed()
        assert (ctx.rank, ctx.world_size, ctx.is_main, ctx.is_distributed) == (0, 1, True, False)
        assert ctx.device.type in ("cpu", "cuda")
