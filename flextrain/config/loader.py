"""Configuration loading utilities."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml
import os

from .base import BaseConfig, FlexTrainConfig
from .training import TrainingConfig
from .distributed import DistributedConfig
from .checkpoint import CheckpointConfig
from .elastic import ElasticConfig


@dataclass
class Config:
    """Complete configuration for FlexTrain."""

    main: FlexTrainConfig = field(default_factory=FlexTrainConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    elastic: ElasticConfig = field(default_factory=ElasticConfig)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "main": self.main.to_dict(),
            "training": self.training.to_dict(),
            "distributed": self.distributed.to_dict(),
            "checkpoint": self.checkpoint.to_dict(),
            "elastic": self.elastic.to_dict(),
        }

    def to_yaml(self, path: Optional[Union[str, Path]] = None) -> str:
        yaml_str = yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)
        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                f.write(yaml_str)
        return yaml_str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        return cls(
            main=FlexTrainConfig.from_dict(data.get("main", {})),
            training=TrainingConfig.from_dict(data.get("training", {})),
            distributed=DistributedConfig.from_dict(data.get("distributed", {})),
            checkpoint=CheckpointConfig.from_dict(data.get("checkpoint", {})),
            elastic=ElasticConfig.from_dict(data.get("elastic", {})),
        )

    def validate(self) -> None:
        self.main.validate()
        self.training.validate()
        self.distributed.validate()
        self.checkpoint.validate()
        self.elastic.validate()


def load_config(path: Union[str, Path]) -> Config:
    """Load configuration from YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    config = Config.from_dict(data or {})
    config.validate()
    return config


def create_default_config() -> Config:
    """Create configuration with default values."""
    return Config()
