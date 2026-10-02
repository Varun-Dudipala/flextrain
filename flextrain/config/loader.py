"""Top-level configuration object and loading utilities."""

from __future__ import annotations

import difflib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import yaml

from .base import BaseConfig, ConfigError, FlexTrainConfig
from .checkpoint import CheckpointConfig
from .distributed import DistributedConfig
from .elastic import ElasticConfig, FaultToleranceConfig
from .training import TrainingConfig

CONFIG_ENV_VAR = "FLEXTRAIN_CONFIG"
OVERRIDES_ENV_VAR = "FLEXTRAIN_OVERRIDES"  # newline-separated section.key=value

_SECTIONS: Dict[str, type] = {
    "main": FlexTrainConfig,
    "training": TrainingConfig,
    "distributed": DistributedConfig,
    "checkpoint": CheckpointConfig,
    "elastic": ElasticConfig,
    "fault_tolerance": FaultToleranceConfig,
}
# Free-form sections passed through untouched for user code (model / dataset hyperparameters).
_FREEFORM = ("model", "data")


@dataclass
class Config:
    """Complete FlexTrain configuration."""

    main: FlexTrainConfig = field(default_factory=FlexTrainConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    elastic: ElasticConfig = field(default_factory=ElasticConfig)
    fault_tolerance: FaultToleranceConfig = field(default_factory=FaultToleranceConfig)
    model: Dict[str, Any] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {name: getattr(self, name).to_dict() for name in _SECTIONS}
        for name in _FREEFORM:
            out[name] = dict(getattr(self, name))
        return out

    def to_yaml(self, path: Optional[Union[str, Path]] = None) -> str:
        yaml_str = yaml.safe_dump(self.to_dict(), default_flow_style=False, sort_keys=False)
        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml_str)
        return yaml_str

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Config:
        data = data or {}
        if not isinstance(data, dict):
            raise ConfigError(f"config root must be a mapping, got {type(data).__name__}")
        known = set(_SECTIONS) | set(_FREEFORM)
        unknown = sorted(set(data) - known)
        if unknown:
            msgs = []
            for key in unknown:
                close = difflib.get_close_matches(key, known, n=1)
                msgs.append(f"'{key}'" + (f" (did you mean '{close[0]}'?)" if close else ""))
            raise ConfigError(f"unknown config section(s) {', '.join(msgs)}")
        kwargs: Dict[str, Any] = {
            name: section_cls.from_dict(data.get(name), section=name) for name, section_cls in _SECTIONS.items()
        }
        for name in _FREEFORM:
            value = data.get(name) or {}
            if not isinstance(value, dict):
                raise ConfigError(f"{name}: expected a mapping, got {type(value).__name__}")
            kwargs[name] = dict(value)
        return cls(**kwargs)

    def copy(self) -> Config:
        return Config.from_dict(self.to_dict())

    def validate(self) -> None:
        """Re-validate every section (useful after mutating fields in code)."""
        for name in _SECTIONS:
            getattr(self, name).validate()

    @property
    def experiment_dir(self) -> Path:
        return Path(self.main.output_dir) / self.main.experiment_name

    def resolved_checkpoint_dir(self) -> str:
        if self.checkpoint.checkpoint_dir:
            return self.checkpoint.checkpoint_dir
        if self.checkpoint.storage_backend != "local":
            return f"{self.main.experiment_name}/checkpoints"
        return str(self.experiment_dir / "checkpoints")


def _parse_override(item: str) -> tuple:
    key, sep, raw = item.partition("=")
    key = key.strip()
    if not sep or not key or any(not part for part in key.split(".")):
        raise ConfigError(f"override must look like section.key=value, got {item!r}")
    return key.split("."), yaml.safe_load(raw) if raw.strip() else None


def apply_overrides(data: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    """Apply ``section.key=value`` overrides to a raw config mapping (values parsed as YAML)."""
    for item in overrides:
        parts, value = _parse_override(item)
        node = data
        for part in parts[:-1]:
            child = node.get(part)
            if child is None:
                child = node[part] = {}
            if not isinstance(child, dict):
                raise ConfigError(f"cannot override {item!r}: {part!r} is not a mapping")
            node = child
        node[parts[-1]] = value
    return data


def load_config(
    path: Optional[Union[str, Path]] = None,
    overrides: Optional[Iterable[str]] = None,
) -> Config:
    """Load and validate a YAML config.

    ``path`` defaults to ``$FLEXTRAIN_CONFIG`` (exported by ``flextrain launch``), in which
    case ``$FLEXTRAIN_OVERRIDES`` (from ``launch --set``) are applied first.
    ``overrides`` are ``section.key=value`` strings applied on top of the file.
    """
    overrides = list(overrides or [])
    if path is None:
        path = os.environ.get(CONFIG_ENV_VAR)
        if not path:
            raise ConfigError(f"no config path given and ${CONFIG_ENV_VAR} is not set")
        env_overrides = [line for line in os.environ.get(OVERRIDES_ENV_VAR, "").splitlines() if line.strip()]
        overrides = env_overrides + overrides
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if overrides:
        apply_overrides(data, overrides)
    return Config.from_dict(data)


def create_default_config() -> Config:
    """Create configuration with default values."""
    return Config()


__all__ = [
    "BaseConfig",
    "Config",
    "CONFIG_ENV_VAR",
    "OVERRIDES_ENV_VAR",
    "apply_overrides",
    "create_default_config",
    "load_config",
]
