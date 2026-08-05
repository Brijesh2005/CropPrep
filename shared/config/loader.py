"""Configuration loading primitives shared across the CropFusion platforms.

Settings in every platform resolve with the same precedence:

1. Environment variables (``<PREFIX><SECTION>__<KEY>``, nested via ``__``).
2. YAML configuration file.
3. Built-in defaults (defined by each platform's pydantic settings).

This module provides the reusable pieces: ``deep_merge``, ``parse_env``,
``apply_case_insensitive`` and ``load_yaml_config``.  They were extracted from
``training.dataset_manager.config`` (and the duplicated copies in the other
training packages) so the backend no longer needs to import private helpers
from ``training``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import yaml
from pydantic import BaseModel

from ..exceptions import ConfigurationError
from ..utils import load_yaml


def deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    result = dict(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, Mapping)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def normalise_key(key: str) -> str:
    """Lower-case + strip a field path segment for case-insensitive matching."""
    return key.lower().replace("-", "_")


def parse_env(env: Mapping[str, str], prefix: str) -> dict[str, Any]:
    """Convert ``<PREFIX><SECTION>__<FIELD>`` env vars into a nested dict.

    Values that look like JSON (``[...]``, ``{...}``, ``true``/``false``,
    integers) are parsed; everything else stays a string.
    """
    overrides: dict[str, Any] = {}
    for raw_key, raw_value in env.items():
        if not raw_key.startswith(prefix):
            continue
        path = raw_key[len(prefix):].split("__")
        path = [normalise_key(part) for part in path if part]
        if not path:
            continue
        value: Any = raw_value
        stripped = raw_value.strip()
        lowered = stripped.lower()
        if lowered in {"true", "false"}:
            value = lowered == "true"
        else:
            try:
                value = json.loads(stripped)
            except (json.JSONDecodeError, TypeError):
                pass
        node = overrides
        for part in path[:-1]:
            node = node.setdefault(part, {})
        node[path[-1]] = value
    return overrides


def apply_case_insensitive(
    data: dict[str, Any], schema: type[BaseModel]
) -> dict[str, Any]:
    """Match config keys to pydantic field names case-insensitively."""
    field_names = {normalise_key(name): name for name in schema.model_fields}
    return {field_names.get(normalise_key(key), key): value for key, value in data.items()}


def load_yaml_config(
    config_path: str | Path | None,
    *,
    env: Mapping[str, str] | None = None,
    prefix: str,
    config_env_var: str = "CONFIG_FILE",
) -> dict[str, Any]:
    """Load a YAML config file and merge environment overrides into a dict.

    Args:
        config_path: Optional YAML file. When None, falls back to the
            environment variable ``config_env_var`` (``<PREFIX>CONFIG_FILE``
            first, then ``CONFIG_FILE``).
        env: Environment mapping; defaults to ``os.environ``.
        prefix: Environment variable prefix (e.g. ``"DM_"``).
        config_env_var: Fallback env var name for the config file path.

    Returns:
        Merged settings mapping (YAML overridden by env), ready for pydantic
        validation.

    Raises:
        ConfigurationError: When the YAML is malformed or not a mapping.
    """
    env_map = dict(os.environ if env is None else env)

    if config_path is None:
        config_path = env_map.get(f"{prefix}CONFIG_FILE") or env_map.get(config_env_var)

    data: dict[str, Any] = {}
    if config_path is not None:
        config_file = Path(config_path)
        if not config_file.exists():
            raise ConfigurationError(
                f"Configuration file not found: {config_file}", detail=str(config_file)
            )
        try:
            raw = load_yaml(config_file)
        except yaml.YAMLError as exc:
            raise ConfigurationError(
                f"Malformed YAML configuration: {exc}", detail=str(config_file)
            ) from exc
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ConfigurationError(
                "Configuration root must be a mapping", detail=str(config_file)
            )
        data = raw

    env_overrides = parse_env(env_map, prefix)
    return deep_merge(data, env_overrides)


# --------------------------------------------------------------------------- #
# Backward-compatible aliases (used by the legacy platform config modules).
# --------------------------------------------------------------------------- #

_parse_env = parse_env
_apply_case_insensitive = apply_case_insensitive
_normalise_key = normalise_key
