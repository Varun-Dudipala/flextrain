"""Single-process Trainer tests (CPU). Multi-rank behaviour lives in tests/integration."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from flextrain import Trainer
from flextrain.config import ConfigError
from flextrain.fault_tolerance import TrainingDivergedError
from flextrain.tracking import RunRegistry
from tests.conftest import NoisyDataset, RegressionDataset, TinyModel, params_equal


def run(make_config, steps=20, model=None, trainer_cls=Trainer, seed=0, dataset=None, tag=None, make_model=None,
        expect_fresh=True, **sections):
    """Train a model from scratch.

    Runs that are compared with each other need two things, or the comparison is vacuous:
    * distinct output directories (pass a ``tag``) - otherwise the second run auto-resumes
      from the first one's final checkpoint and trivially "matches" it;
    * identical initial weights - pass ``make_model`` so the model is built *after* seeding.
    """
    torch.manual_seed(seed)
    if make_model is not None:
        model = make_model()
    cfg = make_config(steps, **sections)
    if tag is not None:
        cfg.main.output_dir += f"-{tag}"
    trainer = trainer_cls(model or TinyModel(), cfg, dataset or RegressionDataset())
    result = trainer.train()
    if tag is not None and expect_fresh:
        assert trainer.resumed_from is None, "tagged runs must start from scratch"
    return trainer, result


class CrashAt(Trainer):
    crash_step = 12

    def _after_step(self):
        stop = super()._after_step()
        if self.global_step == self.crash_step:
            raise RuntimeError("simulated node failure")
        return stop


class TestTraining:
    def test_loss_decreases(self, make_config):
        trainer, result = run(make_config, steps=60, training={"log_interval": 10, "learning_rate": 3e-2})
        losses = [e["loss"] for e in trainer.metrics_logger.read() if "loss" in e]
        assert result.status == "completed" and result.global_step == 60
        assert losses[-1] < 0.5 * losses[0]

    def test_requires_a_stopping_condition(self, make_config):
        cfg = make_config()
        cfg.training.max_steps = None
        with pytest.raises(ConfigError, match="max_steps or training.max_epochs"):
            Trainer(TinyModel(), cfg, RegressionDataset())

    def test_max_epochs(self, make_config):
        cfg = make_config()
        cfg.training.max_steps = None
        cfg.training.max_epochs = 3
        trainer = Trainer(TinyModel(), cfg, RegressionDataset(n=64))
        result = trainer.train()
        assert result.epoch == 3 and result.global_step == 3 * (64 // 8)

    def test_dataset_smaller_than_global_batch(self, make_config):
        with pytest.raises(ValueError, match="fewer than one global batch"):
            Trainer(TinyModel(), make_config(), RegressionDataset(n=4))

    def test_gradient_accumulation_matches_large_batch(self, make_config):
        """batch 4 x accum 2 must equal batch 8 x accum 1 (same samples, same update)."""
        a, _ = run(make_config, make_model=lambda: TinyModel(dropout=0.0), tag="accum",
                   training={"batch_size": 4, "gradient_accumulation_steps": 2})
        b, _ = run(make_config, make_model=lambda: TinyModel(dropout=0.0), tag="large",
                   training={"batch_size": 8, "gradient_accumulation_steps": 1})
        assert a.accumulation_steps == 2 and b.accumulation_steps == 1 and a.global_step == b.global_step == 20
        for p, q in zip(a.model.parameters(), b.model.parameters(), strict=True):
            torch.testing.assert_close(p, q, atol=1e-6, rtol=1e-5)

    def test_activation_checkpointing_does_not_change_results(self, make_config):
        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(8, 8)

            def forward(self, x):
                return torch.relu(self.lin(x))

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.Sequential(Block(), Block())
                self.head = nn.Linear(8, 1)

            def forward(self, x, y):
                out = self.head(self.blocks(x))
                return out, ((out - y) ** 2).mean()

        a, _ = run(make_config, make_model=Net, tag="plain")
        b, _ = run(make_config, make_model=Net, tag="ckpt",
                   distributed={"wrap_modules": ["Block"], "activation_checkpointing": True})
        assert any("_checkpoint_wrapped_module" in n for n, _ in b.model.named_parameters())
        assert a.global_step == b.global_step == 20 and params_equal(a.model, b.model)


class TestExactResume:
    @pytest.mark.parametrize("async_save", [True, False])
    def test_crash_and_resume_is_bitwise_identical(self, make_config, async_save):
        ckpt = {"async_save": async_save}
        straight, _ = run(make_config, checkpoint=ckpt)

        cfg = make_config(checkpoint=ckpt)
        cfg.main.output_dir += "-b"
        torch.manual_seed(0)
        crashed = CrashAt(TinyModel(), cfg, RegressionDataset())
        with pytest.raises(RuntimeError, match="simulated node failure"):
            crashed.train()
        crashed.checkpoint_manager.close(raise_errors=False)

        torch.manual_seed(1234)  # different init: everything must come from the checkpoint
        resumed = Trainer(TinyModel(), cfg.copy(), RegressionDataset())
        result = resumed.train()
        assert Path(resumed.resumed_from).name == "checkpoint_step00000010.pt"
        assert result.global_step == 20 and result.epoch == straight.epoch
        assert params_equal(straight.model, resumed.model)
        assert resumed.run_id == crashed.run_id  # same run identity across restarts

    @pytest.mark.parametrize("crash_workers,resume_workers", [(0, 0), (2, 1)], ids=["in-process", "workers"])
    def test_resume_is_bitwise_identical_with_random_augmentation(self, make_config, crash_workers, resume_workers):
        """Mid-epoch crash with random augmentation. The uninterrupted reference loads data in the
        training process; the crashed run uses 2 worker processes and the resumed run 1. Worker RNG
        state cannot be checkpointed, so the results only match because each sample's randomness
        is seeded from (seed, epoch, index) and is independent of who produces it."""
        data = dict(n=256)  # 32 steps per epoch: the crash at step 12 and the resume are mid-epoch
        straight, _ = run(make_config, dataset=NoisyDataset(**data), training={"num_workers": 0})

        cfg = make_config(training={"num_workers": crash_workers})
        cfg.main.output_dir += "-w"
        torch.manual_seed(0)
        crashed = CrashAt(TinyModel(), cfg, NoisyDataset(**data))
        with pytest.raises(RuntimeError, match="simulated node failure"):
            crashed.train()
        crashed.checkpoint_manager.close(raise_errors=False)

        torch.manual_seed(1234)
        cfg = cfg.copy()
        cfg.training.num_workers = resume_workers
        resumed = Trainer(TinyModel(), cfg, NoisyDataset(**data))
        resumed.train()
        assert Path(resumed.resumed_from).name == "checkpoint_step00000010.pt"
        assert params_equal(straight.model, resumed.model)

    def test_without_per_sample_seeding_worker_augmentation_is_not_reproducible(self, make_config):
        """The control: with the feature off, in-process and worker runs draw different noise."""
        off = {"seed_per_sample": False}
        a, _ = run(make_config, steps=4, dataset=NoisyDataset(), tag="in-process", training={"num_workers": 0, **off})
        b, _ = run(make_config, steps=4, dataset=NoisyDataset(), tag="worker", training={"num_workers": 1, **off})
        assert not params_equal(a.model, b.model)

    def test_resume_from_a_checkpoint_taken_exactly_at_an_epoch_boundary(self, make_config):
        """64 samples at a global batch of 8: step 8 is the last step of epoch 0. The checkpoint is
        taken before the epoch counter advances, with the whole epoch marked consumed; resuming
        must roll over into epoch 1 rather than replay or skip anything."""
        ckpt = {"save_interval_steps": 8}
        straight, _ = run(make_config, tag="epoch-straight", checkpoint=ckpt)
        cfg = crash_then(make_config, "epoch-crash", crash_step=9, checkpoint=ckpt)
        resumed = Trainer(TinyModel(), cfg, RegressionDataset())
        resumed.train()
        assert Path(resumed.resumed_from).name == "checkpoint_step00000008.pt"
        state = torch.load(resumed.resumed_from, weights_only=True)
        assert state["trainer"]["epoch"] == 0 and state["trainer"]["sampler"]["consumed"] == 64
        assert resumed.epoch == straight.epoch and params_equal(straight.model, resumed.model)

    def test_preempt_then_resume_is_bitwise_identical(self, make_config):
        straight, _ = run(make_config)

        class PreemptAt7(Trainer):
            def _after_step(self):
                if self.global_step == 7:
                    self.preemption.request_stop()
                return super()._after_step()

        cfg = make_config()
        cfg.main.output_dir += "-p"
        torch.manual_seed(0)
        first = PreemptAt7(TinyModel(), cfg, RegressionDataset())
        result = first.train()
        assert result.status == "preempted" and result.global_step == 7
        assert Path(result.last_checkpoint).name == "checkpoint_step00000007.pt"
        assert first.registry.get(first.run_id)["status"] == "preempted"

        second = Trainer(TinyModel(), cfg.copy(), RegressionDataset())
        assert second.train().status == "completed"
        assert params_equal(straight.model, second.model)

    def test_explicit_resume_path(self, make_config):
        trainer, _ = run(make_config, steps=10)
        ckpt = trainer.checkpoint_manager.path_for_step(5)
        cfg = make_config(10, checkpoint={"resume_path": ckpt})
        cfg.main.output_dir += "-r"
        resumed = Trainer(TinyModel(), cfg, RegressionDataset())
        resumed.train()
        assert resumed.resumed_from == ckpt

    def test_bad_resume_path_fails_clearly(self, make_config, tmp_path):
        cfg = make_config(checkpoint={"resume_path": str(tmp_path / "missing.pt")})
        with pytest.raises(RuntimeError, match="cannot resume"):
            Trainer(TinyModel(), cfg, RegressionDataset()).train()

    def test_auto_resume_disabled_starts_fresh(self, make_config):
        run(make_config, steps=10)
        trainer, result = run(make_config, steps=10, checkpoint={"auto_resume": False})
        assert trainer.resumed_from is None and result.global_step == 10


def crash_then(make_config, tag, crash_step=12, **sections):
    """Train until a simulated failure at ``crash_step``; return the config to restart with."""
    cfg = make_config(**sections)
    cfg.main.output_dir += f"-{tag}"
    torch.manual_seed(0)
    crashed = CrashAt(TinyModel(), cfg, RegressionDataset())
    crashed.crash_step = crash_step
    with pytest.raises(RuntimeError, match="simulated node failure"):
        crashed.train()
    crashed.checkpoint_manager.close(raise_errors=False)
    return cfg.copy()


def checkpoint_files(cfg):
    return sorted(p.name for p in Path(cfg.resolved_checkpoint_dir()).glob("*.pt"))


class TestLineage:
    """Which checkpoint a run starts from, and which checkpoints it may delete."""

    def test_resume_path_is_a_starting_point_not_a_standing_order(self, make_config):
        """Every restart sees the same config. Going back to resume_path each time would throw
        away everything trained since the branch."""
        source, _ = run(make_config, tag="source")  # 20 steps; checkpoints at 5, 10, 15, 20
        branch_point = source.checkpoint_manager.path_for_step(5)

        cfg = crash_then(make_config, "branch", checkpoint={"resume_path": branch_point})
        restarted = Trainer(TinyModel(), cfg, RegressionDataset())
        result = restarted.train()
        assert Path(restarted.resumed_from).name == "checkpoint_step00000010.pt"
        assert Path(restarted.resumed_from).parent == Path(cfg.resolved_checkpoint_dir())  # its own, not the source's
        assert result.global_step == 20
        assert params_equal(source.model, restarted.model)  # same config from the same point: same weights

    def test_resume_path_inside_the_checkpoint_dir_is_never_pruned(self, make_config):
        first, _ = run(make_config, steps=10, tag="same-dir")  # checkpoints at 5, 10
        branch_point = first.checkpoint_manager.path_for_step(5)
        second, result = run(make_config, steps=20, tag="same-dir", expect_fresh=False,
                             checkpoint={"resume_path": branch_point, "keep_last_n": 1},
                             training={"lr_scheduler": "constant"})
        assert second.resumed_from == branch_point and result.global_step == 20
        assert second.lineage != first.lineage
        # the new lineage kept only its newest checkpoint; the branch point is not its to delete
        assert checkpoint_files(second.config) == ["checkpoint_step00000005.pt", "checkpoint_step00000020.pt"]

    def test_fresh_run_is_not_crowded_out_by_an_older_runs_checkpoints(self, make_config):
        old, _ = run(make_config, steps=20, tag="reused", checkpoint={"keep_last_n": 3})
        new, result = run(make_config, steps=8, tag="reused", checkpoint={"keep_last_n": 3, "auto_resume": False})
        assert new.resumed_from is None and new.lineage != old.lineage
        assert Path(result.last_checkpoint).exists(), "the reported checkpoint must be on storage"
        assert checkpoint_files(new.config) == [f"checkpoint_step{s:08d}.pt" for s in (5, 8, 10, 15, 20)]
        assert [c.step for c in new.checkpoint_manager.committed_checkpoints()] == [5, 8]

    def test_restart_within_one_launch_continues_whatever_the_config_says(self, make_config, monkeypatch):
        monkeypatch.setenv("FLEXTRAIN_LAUNCH_ID", "launch-1")
        cfg = crash_then(make_config, "launch", checkpoint={"auto_resume": False})
        restarted = Trainer(TinyModel(), cfg, RegressionDataset())
        restarted.train()
        assert Path(restarted.resumed_from).name == "checkpoint_step00000010.pt"

        monkeypatch.setenv("FLEXTRAIN_LAUNCH_ID", "launch-2")  # a new launch: auto_resume=false means fresh
        again = Trainer(TinyModel(), cfg.copy(), RegressionDataset())
        again.train()
        assert again.resumed_from is None and again.lineage != restarted.lineage

    def test_lineage_is_kept_across_resume_and_recorded_in_checkpoints(self, make_config):
        cfg = crash_then(make_config, "id")
        resumed = Trainer(TinyModel(), cfg, RegressionDataset())
        resumed.train()
        state = torch.load(resumed.checkpoint_manager.latest_checkpoint(), weights_only=False)
        first = torch.load(resumed.checkpoint_manager.path_for_step(5), weights_only=False)
        assert state["lineage"] == first["lineage"] == resumed.lineage


class TestConfigChange:
    def test_changed_hyperparameters_are_refused_on_continue(self, make_config):
        """Optimizer and schedule state come from the checkpoint, so the new learning rate would
        be silently ignored; the run would look like it used settings it never used."""
        cfg = crash_then(make_config, "lr")
        cfg.training.learning_rate = 0.5
        with pytest.raises(ConfigError, match=r"training\.learning_rate: 0\.01 -> 0\.5"):
            Trainer(TinyModel(), cfg, RegressionDataset()).train()

        cfg.checkpoint.allow_config_change = True
        trainer = Trainer(TinyModel(), cfg, RegressionDataset())
        assert trainer.train().global_step == 20

    def test_operational_settings_may_change_freely(self, make_config):
        cfg = crash_then(make_config, "ops")
        cfg.training.log_interval = 1
        cfg.training.batch_size, cfg.training.gradient_accumulation_steps = 8, 1  # same global batch
        cfg.checkpoint.save_interval_steps = 7
        cfg.checkpoint.keep_last_n = 2
        assert Trainer(TinyModel(), cfg, RegressionDataset()).train().global_step == 20

    def test_training_longer_changes_a_decaying_schedule_but_not_a_constant_one(self, make_config):
        cfg = crash_then(make_config, "cosine")
        cfg.training.max_steps = 40
        with pytest.raises(ConfigError, match=r"training\.max_steps: 20 -> 40"):
            Trainer(TinyModel(), cfg, RegressionDataset()).train()

        cfg = crash_then(make_config, "constant", training={"lr_scheduler": "constant"})
        cfg.training.max_steps = 40
        assert Trainer(TinyModel(), cfg, RegressionDataset()).train().global_step == 40


class TestResumeFailures:
    def test_unreachable_checkpoints_stop_the_job_and_delete_nothing(self, make_config, monkeypatch):
        """A storage outage at restart used to mean: train from step 0, and prune every new
        checkpoint because the unreachable old ones had higher steps."""
        import time

        from flextrain.checkpoint import LocalStorage

        cfg = crash_then(make_config, "outage")
        before = checkpoint_files(cfg)

        def stale_handle(self, path):
            raise OSError("stale NFS file handle")

        monkeypatch.setattr(time, "sleep", lambda s: None)
        monkeypatch.setattr(LocalStorage, "load", stale_handle)
        with pytest.raises(RuntimeError, match="cannot resume: OSError: stale NFS file handle"):
            Trainer(TinyModel(), cfg, RegressionDataset()).train()
        assert checkpoint_files(cfg) == before

    def test_all_checkpoints_corrupted_stops_the_job(self, make_config):
        cfg = crash_then(make_config, "rot")
        for path in Path(cfg.resolved_checkpoint_dir()).glob("*.pt"):
            path.write_bytes(path.read_bytes()[:100])
        with pytest.raises(RuntimeError, match="Refusing to silently start from step 0"):
            Trainer(TinyModel(), cfg, RegressionDataset()).train()

    def test_a_failure_on_any_rank_is_raised_on_every_rank(self, make_config, monkeypatch):
        """Single-process stand-in for the multi-rank case: this rank succeeded, the all-reduce
        says some other rank did not, so this rank must raise too instead of walking on into
        the next collective without it."""
        from flextrain.core import trainer as trainer_module

        trainer = Trainer(TinyModel(), make_config(), RegressionDataset())
        assert trainer._agree(lambda: 42, "saving failed") == 42
        with pytest.raises(OSError, match="disk full"):
            trainer._agree(lambda: (_ for _ in ()).throw(OSError("disk full")), "saving failed")
        monkeypatch.setattr(trainer_module, "all_reduce_tensor", lambda values, op="sum": [1.0])
        with pytest.raises(RuntimeError, match="saving failed on another rank"):
            trainer._agree(lambda: 42, "saving failed")


class TestCheckpointPolicy:
    def ckpt_steps(self, trainer):
        return [c.step for c in trainer.checkpoint_manager.list_checkpoints()]

    def test_interval_and_final(self, make_config):
        trainer, result = run(make_config, steps=12, checkpoint={"save_interval_steps": 5})
        assert self.ckpt_steps(trainer) == [5, 10, 12]
        assert result.last_checkpoint.endswith("checkpoint_step00000012.pt")

    def test_no_final_checkpoint(self, make_config):
        trainer, _ = run(make_config, steps=12, checkpoint={"save_interval_steps": 5, "save_final": False})
        assert self.ckpt_steps(trainer) == [5, 10]

    def test_final_not_duplicated_when_interval_hits_last_step(self, make_config):
        trainer, _ = run(make_config, steps=10, checkpoint={"save_interval_steps": 5})
        assert self.ckpt_steps(trainer) == [5, 10]

    def test_time_based_checkpointing(self, make_config):
        trainer, _ = run(make_config, steps=4, checkpoint={"save_interval_steps": 0, "save_interval_minutes": 1e-9},
                         fault_tolerance={"handle_signals": False})
        assert self.ckpt_steps(trainer) == [1, 2, 3, 4]

    def test_checkpoint_contents(self, make_config):
        trainer, _ = run(make_config, steps=10)
        state = torch.load(trainer.checkpoint_manager.latest_checkpoint(), weights_only=False)
        assert state["step"] == 10 and state["world_size"] == 1 and state["global_batch"] == 8
        assert state["trainer"]["sampler"]["consumed"] == (10 * 8) % 64
        assert len(state["rng"]) == 1 and state["config"]["training"]["max_steps"] == 10
        assert set(state["model"]) == {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"}


class TestDivergenceGuard:
    class SometimesNaN(nn.Module):
        def __init__(self, bad_calls):
            super().__init__()
            self.lin = nn.Linear(8, 1)
            self.calls = 0
            self.bad_calls = bad_calls

        def forward(self, x, y):
            self.calls += 1
            out = self.lin(x)
            loss = ((out - y) ** 2).mean()
            return out, loss * float("nan") if self.calls in self.bad_calls else loss

    def test_nonfinite_steps_are_skipped(self, make_config):
        model = self.SometimesNaN(bad_calls={3, 4})  # micro-batches of optimizer step 2
        trainer, result = run(make_config, steps=6, model=model)
        assert result.status == "completed" and trainer.nonfinite_guard.total_skipped == 1
        assert all(torch.isfinite(p).all() for p in trainer.model.parameters())

    def test_persistent_divergence_raises_and_marks_run_failed(self, make_config):
        model = self.SometimesNaN(bad_calls=set(range(1, 1000)))
        torch.manual_seed(0)
        trainer = Trainer(model, make_config(20, fault_tolerance={"max_consecutive_nonfinite": 2}), RegressionDataset())
        with pytest.raises(TrainingDivergedError):
            trainer.train()
        assert trainer.registry.get(trainer.run_id)["status"] == "failed"

    def with_loss_scaler(self, trainer):
        """The fp16 path normally needs CUDA; the same GradScaler logic runs on CPU."""
        trainer.scaler = torch.amp.GradScaler("cpu")
        return trainer

    def test_guard_counts_steps_the_fp16_scaler_skips(self, make_config):
        torch.manual_seed(0)
        model = self.SometimesNaN(bad_calls={3, 4})  # micro-batches of optimizer step 2
        trainer = self.with_loss_scaler(Trainer(model, make_config(6), RegressionDataset()))
        result = trainer.train()
        assert result.status == "completed" and trainer.nonfinite_guard.total_skipped == 1
        assert all(torch.isfinite(p).all() for p in trainer.model.parameters())
        assert trainer.scaler.get_scale() < 65536.0  # the scaler still backed off after the bad step

    def test_a_step_the_guard_skips_never_reaches_the_optimizer_under_fp16(self, make_config):
        """Finite gradients whose norm overflows: the scaler sees nothing wrong with them and would
        apply the update for a step that is reported as skipped."""
        torch.manual_seed(0)
        trainer = self.with_loss_scaler(Trainer(TinyModel(), make_config(), RegressionDataset()))
        batch = next(iter(trainer.train_loader))
        for _ in range(trainer.accumulation_steps):
            trainer._forward_backward(batch, sync_gradients=True)  # real, finite gradients
        before = [p.detach().clone() for p in trainer.model.parameters()]
        trainer._clip_and_get_grad_norm = lambda: float("inf")
        _, skipped = trainer._optimizer_step()
        assert skipped and trainer.nonfinite_guard.total_skipped == 1
        assert all(torch.equal(a, b) for a, b in zip(before, trainer.model.parameters(), strict=True))

    def test_persistent_divergence_raises_under_fp16_too(self, make_config):
        """Without the guard the scaler would skip every step forever and the job would 'train' on nothing."""
        torch.manual_seed(0)
        model = self.SometimesNaN(bad_calls=set(range(1, 1000)))
        cfg = make_config(20, fault_tolerance={"max_consecutive_nonfinite": 2})
        trainer = self.with_loss_scaler(Trainer(model, cfg, RegressionDataset()))
        with pytest.raises(TrainingDivergedError):
            trainer.train()


class TestTracking:
    def test_metrics_registry_and_run_dir(self, make_config):
        trainer, result = run(make_config, steps=10, training={"log_interval": 5})
        record = RunRegistry().get(result.run_id)
        assert record["status"] == "completed" and record["step"] == 10
        assert record["last_checkpoint_step"] == 10 and record["world_size"] == 1
        events = [json.loads(line) for line in Path(record["metrics_path"]).read_text().splitlines()]
        assert events[0]["event"] == "start" and events[-1]["event"] == "end"
        logged = [e for e in events if "loss" in e]
        assert [e["step"] for e in logged] == [5, 10]
        assert {"lr", "grad_norm", "samples_per_s", "step_time_s"} <= set(logged[0])
        assert (Path(record["run_dir"]) / "config.yaml").exists()

    def test_run_record_carries_the_launch_id(self, make_config, monkeypatch):
        """The launcher finds the runs it started through this field (see elastic.finalize_runs)."""
        monkeypatch.setenv("FLEXTRAIN_LAUNCH_ID", "abc123")
        _, result = run(make_config, steps=2)
        assert RunRegistry().get(result.run_id)["launch_id"] == "abc123"

    def test_run_name_precedence(self, make_config, monkeypatch):
        """config > checkpoint (covered by the resume tests) > the launch's name > the current time."""
        monkeypatch.setenv("FLEXTRAIN_RUN_NAME", "from-launch")
        _, result = run(make_config, steps=2, tag="env")
        assert result.run_id == "default-from-launch"
        _, result = run(make_config, steps=2, tag="cfg", main={"run_name": "from-config"})
        assert result.run_id == "default-from-config"
        monkeypatch.setenv("FLEXTRAIN_RUN_NAME", "../escape")
        with pytest.raises(ConfigError, match="path separators"):
            run(make_config, steps=2, tag="bad")

    def test_evaluation(self, make_config):
        torch.manual_seed(0)
        cfg = make_config(10, training={"eval_interval": 5, "eval_max_batches": 2})
        trainer = Trainer(TinyModel(), cfg, RegressionDataset(), eval_dataset=RegressionDataset(n=32))
        trainer.train()
        evals = [e for e in trainer.metrics_logger.read() if "eval/loss" in e]
        assert [e["step"] for e in evals] == [5, 10]
        assert trainer.model.training  # restored to train mode


class TestCustomization:
    def test_optimizer_and_scheduler_factories(self, make_config):
        torch.manual_seed(0)
        trainer = Trainer(TinyModel(), make_config(4), RegressionDataset(),
                          optimizer=lambda params: torch.optim.SGD(params, lr=0.1),
                          lr_scheduler=lambda opt: torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.5))
        trainer.train()
        assert isinstance(trainer.optimizer, torch.optim.SGD)
        assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.1 * 0.25)

    def test_optimizer_instance(self, make_config):
        model = TinyModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        assert Trainer(model, make_config(), RegressionDataset(), optimizer=opt).optimizer is opt

    def test_no_weight_decay_on_biases(self, make_config):
        trainer = Trainer(TinyModel(), make_config(), RegressionDataset())
        groups = {g["weight_decay"]: g["params"] for g in trainer.optimizer.param_groups}
        assert all(p.ndim >= 2 for p in groups[trainer.config.training.weight_decay])
        assert all(p.ndim == 1 for p in groups[0.0])

    def test_weight_decay_grouping_is_by_name_through_wrappers(self, make_config):
        """Wrappers (activation checkpointing here; DDP/FSDP in production) rename parameters;
        FSDP also flattens their shapes. Grouping must still follow the original tensors."""

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(8, 8)

            def forward(self, x):
                return self.lin(x)

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.block = Block()
                self.head = nn.Linear(8, 1)

            def forward(self, x, y):
                out = self.head(self.block(x))
                return out, ((out - y) ** 2).mean()

        cfg = make_config(distributed={"wrap_modules": ["Block"], "activation_checkpointing": True})
        trainer = Trainer(Net(), cfg, RegressionDataset())
        assert any("_checkpoint_wrapped_module" in n for n, _ in trainer.model.named_parameters())
        decay = {id(p) for g in trainer.optimizer.param_groups if g["weight_decay"] > 0 for p in g["params"]}
        expected = {id(p) for p in trainer.model.parameters() if p.ndim == 2}
        assert decay == expected and len(expected) == 2

    @pytest.mark.parametrize("name,canonical", [
        ("module.fc.weight", "fc.weight"),
        ("_fsdp_wrapped_module.h.0._fsdp_wrapped_module.attn.weight", "h.0.attn.weight"),
        ("h.0._checkpoint_wrapped_module.mlp.bias", "h.0.mlp.bias"),
        ("_orig_mod.wte.weight", "wte.weight"),
    ])
    def test_canonical_parameter_names(self, name, canonical):
        from flextrain.core.trainer import _canonical_name

        assert _canonical_name(name) == canonical

    # warmup_steps=2, max_steps=12: step 7 is half-way through the decay, step 12 its end.
    @pytest.mark.parametrize("kind,checks", [
        ("cosine", {0: 0.5, 1: 1.0, 2: 1.0, 7: 0.55, 12: 0.1}),
        ("linear", {0: 0.5, 1: 1.0, 7: 0.55, 12: 0.1}),
        ("constant", {0: 0.5, 1: 1.0, 12: 1.0}),
    ])
    def test_lr_schedule(self, make_config, kind, checks):
        cfg = make_config(12, training={"lr_scheduler": kind, "warmup_steps": 2, "min_lr_ratio": 0.1,
                                        "learning_rate": 1.0})
        trainer = Trainer(TinyModel(), cfg, RegressionDataset())
        lam = trainer.lr_scheduler.lr_lambdas[0]
        for step, expected in checks.items():
            assert lam(step) == pytest.approx(expected, abs=1e-6), (kind, step)

    @pytest.mark.parametrize("output", [
        torch.tensor(2.0), {"loss": torch.tensor(2.0)}, SimpleNamespace(loss=torch.tensor(2.0)),
        (torch.zeros(2), torch.tensor(2.0)),
    ], ids=["scalar", "dict", "attr", "tuple"])
    def test_loss_extraction_conventions(self, make_config, output):
        trainer = Trainer(TinyModel(), make_config(), RegressionDataset())
        trainer.model = lambda **kw: output
        assert trainer.compute_loss({"x": None}).item() == 2.0

    def test_batches_keep_their_container_type_on_the_way_to_the_device(self):
        from collections import namedtuple

        from flextrain.core.trainer import _move_to_device

        Pair = namedtuple("Pair", "x y")
        cpu = torch.device("cpu")
        moved = _move_to_device(Pair(torch.ones(2), [torch.zeros(1), "label"]), cpu)
        assert isinstance(moved, Pair) and isinstance(moved.y, list) and moved.y[1] == "label"
        assert isinstance(_move_to_device((torch.ones(1), 3), cpu), tuple)

    def test_bf16_falls_back_to_fp16_on_gpus_that_only_emulate_it(self, make_config, monkeypatch):
        """torch reports bf16 as 'supported' on a T4 by counting slow emulation; only native support
        should keep bf16, otherwise the fallback to fp16 with loss scaling never happens."""
        asked = []

        def is_bf16_supported(including_emulation=True):
            asked.append(including_emulation)
            return including_emulation  # an older GPU: emulated yes, native no

        monkeypatch.setattr(torch.cuda, "is_bf16_supported", is_bf16_supported)
        trainer = Trainer(TinyModel(), make_config(), RegressionDataset())
        trainer.device = torch.device("cuda")
        assert trainer._resolve_amp_dtype() == torch.float16 and asked == [False]

    def test_unsupported_model_output(self, make_config):
        trainer = Trainer(TinyModel(), make_config(), RegressionDataset())
        trainer.model = lambda **kw: "nope"
        with pytest.raises(TypeError, match="compute_loss"):
            trainer.compute_loss({})
