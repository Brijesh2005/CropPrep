"""Environment variable parsing helpers."""

from __future__ import annotations

import json
import os
from typing import Any, Mapping


def get_env(name: str, default: str | None = None) -> str | None:
    """Read an environment variable."""
    return os.environ.get(name, default)  # type: ignore[arg-type]


def parse_bool(value: Any) -> bool:
    """Parse a value as a boolean, accepting common textual spellings."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "on", "y"}:
        return True
    return False


def parse_int(value: Any, default: int | None = None) -> int | None:
    """Parse a value as an int, falling back to ``default``."""
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def parse_json(value: Any) -> Any:
    """Parse a JSON string; returns the raw value when it is not JSON."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def env_map_of(prefix: str, env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return every env var whose name starts with ``prefix`` (sans prefix)."""
    source = os.environ if env is None else env
    return {
        key[len(prefix):]: value
        for key, value in source.items()
        if key.startswith(prefix) and len(key) > len(prefix)
    }
