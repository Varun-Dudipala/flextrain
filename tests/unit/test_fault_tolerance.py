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
