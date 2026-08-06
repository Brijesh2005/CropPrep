"""Feature-engineering configuration.

Settings resolve with the same precedence as the rest of the platform:
environment variables (prefix ``FE_``, nesting separated by ``__``) override a
YAML file (``FE_CONFIG_FILE`` / ``--config``) which overrides built-in
defaults. Every field is validated by pydantic.

Key options::

    FE_TABULAR__ENABLED=true
    FE_IMAGE__EXTRACT_PATCH_STATS=false
    FE_IMAGE__MAX_DATES=8
    FE_TEMPORAL__ENABLED=true
    FE_PREFIXES=true
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field

from shared.config import apply_case_insensitive, deep_merge, parse_env
from .exceptions import FeatureConfigError

ENV_PREFIX = "FE_"


class TabularFeatureConfig(BaseModel):
    """Settings for the tabular / location feature builder."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    #: Include query-point location features (lon/lat/distance/admin).
    include_location: bool = True
    #: Include crop / yield training labels in the feature row.
    include_labels: bool = True
    #: Skip tabular ``fields`` when True (keeps only resolved labels).
    include_fields: bool = True
    #: Raw tabular columns treated as training labels and therefore excluded
    #: from the generic feature fields (the resolved ``crop`` / ``yield_value``
    #: columns are controlled separately by ``include_labels``).
    label_columns: list[str] = Field(
        default_factory=lambda: ["crop", "yield", "yield_value", "yield_kg"]
    )


class ImageFeatureConfig(BaseModel):
    """Settings for the image feature builder."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    #: Also extract per-date patch statistics (requires an extractor).
    extract_patch_stats: bool = False
    #: Maximum number of dates used for patch statistics.
    max_dates: int = Field(default=8, ge=1, le=64)
    #: Patch edge (px) requested from the extractor.
    patch_size: int = Field(default=128, ge=8, le=2048)


class TemporalFeatureConfig(BaseModel):
    """Settings for the temporal / sequence feature builder."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    #: Include string-formatted date columns (iso dates) in the row.
    include_dates: bool = True


class FeatureEngineeringConfig(BaseModel):
    """Root feature-engineering settings object."""

    model_config = ConfigDict(extra="forbid")

    tabular: TabularFeatureConfig = Field(default_factory=TabularFeatureConfig)
    image: ImageFeatureConfig = Field(default_factory=ImageFeatureConfig)
    temporal: TemporalFeatureConfig = Field(default_factory=TemporalFeatureConfig)
    #: Prefix every feature with its modality (``tab.*``, ``img.*``, ``tmp.*``).
    prefixes: bool = True


def load_feature_engineering_config(
    config_path: str | Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> FeatureEngineeringConfig:
    """Load and validate feature-engineering settings (env > YAML > defaults).

    Args:
        config_path: Optional YAML file (falls back to ``FE_CONFIG_FILE``).
        env: Environment mapping (defaults to ``os.environ``).

    Raises:
        FeatureConfigError: For malformed YAML or invalid values.
    """
    env_map = dict(os.environ if env is None else env)
    if config_path is None:
        env_config = env_map.get("FE_CONFIG_FILE")
        config_path = env_config or None

    data: dict[str, Any] = {}
    if config_path is not None:
        config_file = Path(config_path)
        if not config_file.exists():
            raise FeatureConfigError(
                f"Feature-engineering config file not found: {config_file}",
                detail=str(config_file),
            )
        try:
            raw = yaml.safe_load(config_file.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise FeatureConfigError(
                f"Malformed feature-engineering YAML: {exc}",
                detail=str(config_file),
            ) from exc
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise FeatureConfigError("Feature-engineering config root must be a mapping")
        data = raw

    merged = deep_merge(data, parse_env(env_map, prefix=ENV_PREFIX))
    merged = apply_case_insensitive(merged, FeatureEngineeringConfig)
    try:
        return FeatureEngineeringConfig.model_validate(merged)
    except Exception as exc:  # pydantic.ValidationError
        raise FeatureConfigError(f"Invalid feature-engineering configuration: {exc}") from exc


def save_feature_engineering_template(path: str | Path) -> Path:
    """Write an annotated YAML template of the default configuration."""
    template = {
        "tabular": {"enabled": True, "include_location": True,
                    "include_labels": True, "include_fields": True,
                    "label_columns": ["crop", "yield", "yield_value", "yield_kg"]},
        "image": {"enabled": True, "extract_patch_stats": False,
                  "max_dates": 8, "patch_size": 128},
        "temporal": {"enabled": True, "include_dates": True},
        "prefixes": True,
    }
    out = Path(path)
    out.write_text(yaml.safe_dump(template, sort_keys=False), encoding="utf-8")
    return out
