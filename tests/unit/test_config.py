"""Unit tests for the configuration system."""

import pytest
import yaml

from flextrain.config import (
    CheckpointConfig,
    Config,
    ConfigError,
    DistributedConfig,
    ElasticConfig,
    FaultToleranceConfig,
    TrainingConfig,
    apply_overrides,
    load_config,
)


def write_yaml(tmp_path, text: str):
    path = tmp_path / "config.yaml"
    path.write_text(text)
    return path


class TestTypeCoercion:
    def test_yaml_scientific_notation_loads_as_float(self, tmp_path):
        # PyYAML parses `1e-4` as the *string* "1e-4"; it must still become a float.
        cfg = load_config(write_yaml(tmp_path, "training:\n  learning_rate: 1e-4\n  max_steps: 1e3\n"))
        assert cfg.training.learning_rate == pytest.approx(1e-4)
        assert isinstance(cfg.training.learning_rate, float)
        assert cfg.training.max_steps == 1000 and isinstance(cfg.training.max_steps, int)

    @pytest.mark.parametrize("value,expected", [("true", True), ("no", False), (True, True), ("1", True)])
    def test_bool_coercion(self, value, expected):
        assert TrainingConfig.from_dict({"pin_memory": value}).pin_memory is expected

    @pytest.mark.parametrize("field,value", [
        ("batch_size", "abc"), ("batch_size", 2.5), ("batch_size", True),
        ("learning_rate", "fast"), ("pin_memory", "maybe"), ("max_steps", [1]),
    ])
    def test_bad_types_raise_with_field_name(self, field, value):
        with pytest.raises(ConfigError, match=f"training.{field}"):
            TrainingConfig.from_dict({field: value}, section="training")

    def test_optional_accepts_null(self):
        assert TrainingConfig.from_dict({"max_steps": None}).max_steps is None

    def test_list_field(self):
        cfg = DistributedConfig.from_dict({"wrap_modules": ["Block"], "activation_checkpointing": True})
        assert cfg.wrap_modules == ["Block"]
        with pytest.raises(ConfigError):
            DistributedConfig.from_dict({"wrap_modules": "Block"})


class TestStrictKeys:
    def test_unknown_key_suggests_correction(self):
        with pytest.raises(ConfigError, match="did you mean 'learning_rate'"):
            Config.from_dict({"training": {"learning_rat": 0.1}})

    def test_unknown_section_suggests_correction(self):
        with pytest.raises(ConfigError, match="did you mean 'training'"):
            Config.from_dict({"trainng": {}})

    def test_freeform_sections_accept_anything(self):
        cfg = Config.from_dict({"model": {"size": "tiny", "nested": {"a": 1}}, "data": {"seq_len": 128}})
        assert cfg.model["nested"]["a"] == 1 and cfg.data["seq_len"] == 128


class TestValidation:
    @pytest.mark.parametrize("kwargs", [
        {"batch_size": 0}, {"learning_rate": -1}, {"gradient_accumulation_steps": 0}, {"optimizer": "lion"},
        {"lr_scheduler": "step"}, {"beta1": 1.0}, {"min_lr_ratio": 2}, {"max_steps": 0},
        {"max_steps": 10, "warmup_steps": 20}, {"global_batch_size": 2, "batch_size": 4},
        {"log_interval": 0}, {"max_grad_norm": -1},
    ])
    def test_training_rejects(self, kwargs):
        with pytest.raises(ConfigError):
            TrainingConfig(**kwargs)

    def test_distributed_rejects_unknown_strategy(self):
        with pytest.raises(ConfigError):
            DistributedConfig(strategy="horovod")

    def test_distributed_normalizes_case(self):
        cfg = DistributedConfig(strategy="FSDP", sharding_strategy="full_shard", mixed_precision_dtype="BF16")
        assert (cfg.strategy, cfg.sharding_strategy, cfg.mixed_precision_dtype) == ("fsdp", "FULL_SHARD", "bf16")

    def test_activation_checkpointing_requires_wrap_modules(self):
        with pytest.raises(ConfigError, match="wrap_modules"):
            DistributedConfig(activation_checkpointing=True)

    def test_cloud_backend_requires_bucket(self):
        with pytest.raises(ConfigError, match="bucket_name"):
            CheckpointConfig(storage_backend="s3")
        assert CheckpointConfig(storage_backend="s3", bucket_name="b").bucket_name == "b"

    @pytest.mark.parametrize("kwargs", [{"keep_last_n": 0}, {"max_pending_saves": 0},
                                        {"storage_backend": "ftp"}, {"save_interval_minutes": 0}])
    def test_checkpoint_rejects(self, kwargs):
        with pytest.raises(ConfigError):
            CheckpointConfig(**kwargs)

    @pytest.mark.parametrize("kwargs", [{"min_nodes": 0}, {"min_nodes": 4, "max_nodes": 2},
                                        {"rdzv_endpoint": "nohost"}, {"max_restarts": -1}])
    def test_elastic_rejects(self, kwargs):
        with pytest.raises(ConfigError):
            ElasticConfig(**kwargs)

    def test_fault_tolerance_rejects_unknown_signal(self):
        with pytest.raises(ConfigError, match="SIGFOO"):
            FaultToleranceConfig(signals=["SIGFOO"])

    def test_run_name_cannot_escape_output_dir(self):
        with pytest.raises(ConfigError):
            Config.from_dict({"main": {"run_name": "../evil"}})

    def test_log_level_validated(self):
        with pytest.raises(ConfigError):
            Config.from_dict({"main": {"log_level": "LOUD"}})


class TestBatchGeometry:
    def test_fixed_accumulation(self):
        tc = TrainingConfig(batch_size=8, gradient_accumulation_steps=4)
        assert tc.accumulation_steps_for(1) == 4
        assert tc.global_batch_for(2) == 64

    @pytest.mark.parametrize("world_size,accum", [(1, 8), (2, 4), (4, 2), (8, 1)])
    def test_global_batch_is_preserved_across_world_sizes(self, world_size, accum):
        tc = TrainingConfig(batch_size=4, global_batch_size=32)
        assert tc.accumulation_steps_for(world_size) == accum
        assert tc.global_batch_for(world_size) == 32

    def test_indivisible_global_batch_rounds_to_nearest(self):
        tc = TrainingConfig(batch_size=4, global_batch_size=32)
        assert tc.accumulation_steps_for(3) == 3  # 32 / 12 = 2.67 -> 3, effective 36
        assert tc.global_batch_for(3) == 36


class TestConfigIO:
    def test_defaults(self):
        cfg = Config()
        assert cfg.training.batch_size == 32 and cfg.distributed.strategy == "ddp"

    def test_yaml_round_trip_is_lossless(self, tmp_path):
        cfg = Config()
        cfg.training.learning_rate = 3e-4
        cfg.distributed.wrap_modules = ["GPT2Block"]
        cfg.model = {"size": "tiny"}
        path = tmp_path / "c.yaml"
        cfg.to_yaml(path)
        assert load_config(path).to_dict() == cfg.to_dict()

    def test_copy_is_deep(self):
        cfg = Config()
        clone = cfg.copy()
        clone.training.batch_size = 1
        assert cfg.training.batch_size == 32

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nope.yaml")

    def test_empty_file_gives_defaults(self, tmp_path):
        assert load_config(write_yaml(tmp_path, "")).training.batch_size == 32

    def test_checkpoint_dir_defaults_to_experiment_dir(self):
        cfg = Config.from_dict({"main": {"output_dir": "/runs", "experiment_name": "exp"}})
        assert cfg.resolved_checkpoint_dir() == "/runs/exp/checkpoints"
        cfg.checkpoint.checkpoint_dir = "/elsewhere"
        assert cfg.resolved_checkpoint_dir() == "/elsewhere"

    def test_shipped_example_config_is_valid(self):
        from pathlib import Path

        example = Path(__file__).resolve().parents[2] / "examples" / "gpt2_training" / "config.yaml"
        cfg = load_config(example)
        assert cfg.training.learning_rate == pytest.approx(1e-3)


class TestOverrides:
    def test_overrides_parse_yaml_values(self):
        data = apply_overrides({}, ["training.learning_rate=3e-4", "training.max_steps=10",
                                    "distributed.wrap_modules=[A, B]", "model.size=tiny"])
        cfg = Config.from_dict(data)
        assert cfg.training.learning_rate == pytest.approx(3e-4)
        assert cfg.distributed.wrap_modules == ["A", "B"] and cfg.model == {"size": "tiny"}

    @pytest.mark.parametrize("bad", ["nokey", "=1", "training.=1", ".x=1"])
    def test_malformed_override(self, bad):
        with pytest.raises(ConfigError):
            apply_overrides({}, [bad])

    def test_load_config_from_env_with_env_overrides(self, tmp_path, monkeypatch):
        path = write_yaml(tmp_path, yaml.safe_dump({"training": {"batch_size": 8}}))
        monkeypatch.setenv("FLEXTRAIN_CONFIG", str(path))
        monkeypatch.setenv("FLEXTRAIN_OVERRIDES", "training.batch_size=16\ntraining.max_steps=5")
        cfg = load_config(overrides=["training.max_steps=7"])
        assert cfg.training.batch_size == 16 and cfg.training.max_steps == 7

    def test_load_config_without_path_or_env(self):
        with pytest.raises(ConfigError, match="FLEXTRAIN_CONFIG"):
            load_config()
