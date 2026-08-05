"""YAML serialization helpers (safe load/dump + primitives sanitisation)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def yaml_safe(value: Any) -> Any:
    """Convert Path / non-primitive values so YAML serialization succeeds.

    Replaces the private ``_yaml_safe`` copies that were duplicated across
    the training packages and the backend config module.  Objects exposing a
    scalar ``.item()`` method (torch tensors, numpy scalars) are converted via
    that method first.
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): yaml_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [yaml_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return yaml_safe(value.item())
        except Exception:
            return str(value)
    return value


def load_yaml(path: str | Path) -> Any:
    """Safely load a YAML file, returning None when it is missing."""
    file_path = Path(path)
    if not file_path.exists():
        return None
    with open(file_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def dump_yaml(data: Any, *, sort_keys: bool = False, allow_unicode: bool = True) -> str:
    """Serialize ``data`` to a YAML string (Path / enums sanitised first)."""
    return yaml.safe_dump(yaml_safe(data), sort_keys=sort_keys, allow_unicode=allow_unicode)


def write_yaml(path: str | Path, data: Any, *, sort_keys: bool = False) -> Path:
    """Write ``data`` to a YAML file and return the path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dump_yaml(data, sort_keys=sort_keys), encoding="utf-8")
    return out


def env_config_path(env_var: str = "CONFIG_FILE") -> str | None:
    """Return the config file path from the environment, if set."""
    return os.environ.get(env_var) or None
