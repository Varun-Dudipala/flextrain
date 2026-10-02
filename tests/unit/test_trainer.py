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
from tests.conftest import RegressionDataset, TinyModel, params_equal


def run(make_config, steps=20, model=None, trainer_cls=Trainer, seed=0, dataset=None, **sections):
    torch.manual_seed(seed)
    trainer = trainer_cls(model or TinyModel(), make_config(steps, **sections), dataset or RegressionDataset())
    return trainer, trainer.train()


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
        a, _ = run(make_config, model=TinyModel(dropout=0.0),
                   training={"batch_size": 4, "gradient_accumulation_steps": 2})
        b, _ = run(make_config, model=TinyModel(dropout=0.0),
                   training={"batch_size": 8, "gradient_accumulation_steps": 1})
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

        a, _ = run(make_config, model=Net())
        b, _ = run(make_config, model=Net(),
                   distributed={"wrap_modules": ["Block"], "activation_checkpointing": True})
        assert params_equal(a.model, b.model)


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

    def test_unsupported_model_output(self, make_config):
        trainer = Trainer(TinyModel(), make_config(), RegressionDataset())
        trainer.model = lambda **kw: "nope"
        with pytest.raises(TypeError, match="compute_loss"):
            trainer.compute_loss({})
