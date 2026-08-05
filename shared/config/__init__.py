"""Shared configuration primitives.

Provides the precedence resolution helpers (env > YAML > defaults) used by
every platform's settings loader.  See :mod:`shared.config.loader` for the
implementation.
"""

from __future__ import annotations

from .loader import (
    _apply_case_insensitive,
    _normalise_key,
    _parse_env,
    apply_case_insensitive,
    deep_merge,
    load_yaml_config,
    normalise_key,
    parse_env,
)

__all__ = [
    "_apply_case_insensitive",
    "_normalise_key",
    "_parse_env",
    "apply_case_insensitive",
    "deep_merge",
    "load_yaml_config",
    "normalise_key",
    "parse_env",
]
