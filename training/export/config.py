"""Dataset-export configuration.

Settings resolve with the same precedence as the rest of the platform:
environment variables (prefix ``EX_``, nesting separated by ``__``) override a
YAML file (``EX_CONFIG_FILE`` / ``--config``) which overrides built-in
defaults. Every field is validated by pydantic.

Key options::

    EX_OUTPUT_DIR=data/out/datasets
    EX_FORMATS=["json","parquet","torch"]
    EX_PREFIX=cropfusion
    EX_INCLUDE_QUALITY=true
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from shared.config import apply_case_insensitive, deep_merge, parse_env
from .exceptions import ExportConfigError

ENV_PREFIX = "EX_"

SUPPORTED_FORMATS = ("json", "jsonl", "parquet", "torch")


class ExportConfig(BaseModel):
    """Settings for exporting a generated training dataset."""

    model_config = ConfigDict(extra="forbid")

    #: Directory where artifacts are written.
    output_dir: str = "data/out/datasets"
    #: Requested artifact formats (subset of ``json``, ``jsonl``, ``parquet``,
    #: ``torch``).
    formats: list[str] = Field(default_factory=lambda: ["json", "parquet"])
    #: File-name prefix for every artifact (e.g. ``cropfusion.json``).
    prefix: str = "cropfusion"
    #: Attach corpus metadata (sample id, year, season, quality score) to rows.
    include_meta: bool = True
    #: Include the ``quality_score`` meta column.
    include_quality: bool = True
    #: Write a ``manifest.json`` describing every artifact.
    write_manifest: bool = True

    @field_validator("formats")
    @classmethod
    def _validate_formats(cls, value: list[str]) -> list[str]:
        unknown = [f for f in value if f not in SUPPORTED_FORMATS]
        if unknown:
            raise ValueError(f"Unsupported export format(s): {', '.join(unknown)}")
        return value


def load_export_config(
    config_path: str | Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> ExportConfig:
    """Load and validate export settings (env > YAML > defaults).

    Args:
        config_path: Optional YAML file (falls back to ``EX_CONFIG_FILE``).
        env: Environment mapping (defaults to ``os.environ``).

    Raises:
        ExportConfigError: For malformed YAML or invalid values.
    """
    env_map = dict(os.environ if env is None else env)
    if config_path is None:
        config_path = env_map.get("EX_CONFIG_FILE") or None

    data: dict[str, Any] = {}
    if config_path is not None:
        config_file = Path(config_path)
        if not config_file.exists():
            raise ExportConfigError(
                f"Export config file not found: {config_file}",
                detail=str(config_file),
            )
        try:
            raw = yaml.safe_load(config_file.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ExportConfigError(
                f"Malformed export YAML: {exc}",
                detail=str(config_file),
            ) from exc
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ExportConfigError("Export config root must be a mapping")
        data = raw

    merged = deep_merge(data, parse_env(env_map, prefix=ENV_PREFIX))
    merged = apply_case_insensitive(merged, ExportConfig)
    try:
        return ExportConfig.model_validate(merged)
    except Exception as exc:  # pydantic.ValidationError
        raise ExportConfigError(f"Invalid export configuration: {exc}") from exc


def save_export_template(path: str | Path) -> Path:
    """Write an annotated YAML template of the default configuration."""
    template = {
        "output_dir": "data/out/datasets",
        "formats": ["json", "parquet"],
        "prefix": "cropfusion",
        "include_meta": True,
        "include_quality": True,
        "write_manifest": True,
    }
    out = Path(path)
    out.write_text(yaml.safe_dump(template, sort_keys=False), encoding="utf-8")
    return out
