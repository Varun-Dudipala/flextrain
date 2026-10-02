"""Base configuration classes with typed, strict (de)serialization.

Every config section is a dataclass. ``from_dict`` is deliberately strict:

* unknown keys raise (a typo such as ``learning_rat`` must not be silently ignored),
* values are coerced to the annotated type, so YAML quirks like ``1e-4`` (which
  PyYAML parses as the *string* ``"1e-4"``) still load as floats,
* anything that cannot be coerced raises a ``ConfigError`` naming the field.
"""

from __future__ import annotations

import difflib
import json
import typing
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar, Union

import yaml

T = TypeVar("T", bound="BaseConfig")

_BOOL_STRINGS = {"true": True, "yes": True, "on": True, "1": True,
                 "false": False, "no": False, "off": False, "0": False}


class ConfigError(ValueError):
    """Raised when a configuration value is missing, malformed or invalid."""


def _coerce(value: Any, annotation: Any, name: str) -> Any:
    """Coerce ``value`` to ``annotation`` or raise ``ConfigError``."""
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    if annotation is Any:
        return value

    if origin is Union:  # Optional[X] == Union[X, None]
        if value is None and type(None) in args:
            return None
        non_none = [a for a in args if a is not type(None)]
        errors = []
        for candidate in non_none:
            try:
                return _coerce(value, candidate, name)
            except ConfigError as e:
                errors.append(str(e))
        raise ConfigError(errors[0] if len(errors) == 1 else f"{name}: {value!r} matches none of {non_none}")

    if value is None:
        raise ConfigError(f"{name}: value is required (got null)")

    if origin in (list, tuple):
        if isinstance(value, str) or not isinstance(value, (list, tuple)):
            raise ConfigError(f"{name}: expected a list, got {value!r}")
        item_type = args[0] if args else Any
        return [_coerce(v, item_type, f"{name}[{i}]") for i, v in enumerate(value)]

    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{name}: expected a mapping, got {value!r}")
        return dict(value)

    if annotation is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in _BOOL_STRINGS:
            return _BOOL_STRINGS[value.strip().lower()]
        raise ConfigError(f"{name}: expected a boolean, got {value!r}")

    if annotation is int:
        if isinstance(value, bool):
            raise ConfigError(f"{name}: expected an integer, got {value!r}")
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            try:
                as_float = float(value)
            except ValueError:
                pass
            else:
                if as_float.is_integer():
                    return int(as_float)
        raise ConfigError(f"{name}: expected an integer, got {value!r}")

    if annotation is float:
        if isinstance(value, bool):
            raise ConfigError(f"{name}: expected a number, got {value!r}")
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)  # handles YAML's "1e-4" string
            except ValueError:
                pass
        raise ConfigError(f"{name}: expected a number, got {value!r}")

    if annotation is str:
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            return str(value)
        raise ConfigError(f"{name}: expected a string, got {value!r}")

    return value


@dataclass
class BaseConfig:
    """Base configuration class with serialization and validation support."""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: Optional[Union[str, Path]] = None) -> str:
        yaml_str = yaml.safe_dump(self.to_dict(), default_flow_style=False, sort_keys=False)
        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml_str)
        return yaml_str

    def to_json(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def section_name(cls) -> str:
        return cls.__name__

    @classmethod
    def from_dict(cls: Type[T], data: Optional[Dict[str, Any]], section: Optional[str] = None) -> T:
        """Build a config from a mapping, coercing types and rejecting unknown keys."""
        section = section or cls.section_name()
        data = data or {}
        if not isinstance(data, dict):
            raise ConfigError(f"{section}: expected a mapping, got {type(data).__name__}")

        hints = typing.get_type_hints(cls)
        valid = {f.name for f in fields(cls) if f.init}
        unknown = sorted(set(data) - valid)
        if unknown:
            hints_msg = []
            for key in unknown:
                close = difflib.get_close_matches(key, valid, n=1)
                hints_msg.append(f"'{key}'" + (f" (did you mean '{close[0]}'?)" if close else ""))
            raise ConfigError(f"{section}: unknown key(s) {', '.join(hints_msg)}")

        kwargs = {k: _coerce(v, hints[k], f"{section}.{k}") for k, v in data.items()}
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls: Type[T], path: Union[str, Path]) -> T:
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    def validate(self) -> None:
        """Validate configuration values. Subclasses raise ``ConfigError``."""

    def __post_init__(self) -> None:
        self.validate()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def _one_of(name: str, value: str, choices) -> None:
    _require(value in choices, f"{name} must be one of {sorted(choices)}, got {value!r}")


@dataclass
class FlexTrainConfig(BaseConfig):
    """Run-level settings (the ``main`` section)."""

    experiment_name: str = "default"
    run_name: Optional[str] = None
    output_dir: str = "./outputs"
    seed: int = 42
    log_level: str = "INFO"
    log_to_file: bool = True

    @classmethod
    def section_name(cls) -> str:
        return "main"

    def validate(self) -> None:
        self.log_level = self.log_level.upper()
        _one_of("main.log_level", self.log_level, {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})
        _require(bool(self.experiment_name.strip()), "main.experiment_name must be non-empty")
        for name, value in (("experiment_name", self.experiment_name), ("run_name", self.run_name)):
            if value is not None:
                _require("/" not in value and "\\" not in value,
                         f"main.{name} must not contain path separators, got {value!r}")
