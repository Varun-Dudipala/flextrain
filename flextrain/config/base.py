"""Base configuration classes with serialization support."""

from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Any, Dict, Optional, TypeVar, Type
import yaml
import json

T = TypeVar("T", bound="BaseConfig")


@dataclass
class BaseConfig:
    """Base configuration class with serialization support."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def to_yaml(self, path: Optional[Path] = None) -> str:
        """Save configuration to YAML file or return as string."""
        yaml_str = yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)
        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                f.write(yaml_str)
        return yaml_str

    def to_json(self, path: Path) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Create configuration from dictionary."""
        valid_fields = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)

    @classmethod
    def from_yaml(cls: Type[T], path: Path) -> T:
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data or {})

    def validate(self) -> None:
        """Validate configuration values."""
        pass

    def __post_init__(self) -> None:
        """Run validation after initialization."""
        self.validate()


@dataclass
class FlexTrainConfig(BaseConfig):
    """Master configuration."""
    experiment_name: str = "default"
    run_name: Optional[str] = None
    output_dir: str = "./outputs"
    seed: int = 42
    log_level: str = "INFO"
    log_to_file: bool = True

    def validate(self) -> None:
        valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level.upper() not in valid_log_levels:
            raise ValueError(f"log_level must be one of {valid_log_levels}")
