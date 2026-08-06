"""Configuration for the inference package (Phase R5).

Everything is configurable through YAML (or ``INF_*`` env vars), mirroring the
resolution order of the other CropFusion packages:

    env (``INF_<SECTION>__<KEY>``) > YAML (``INF_CONFIG_FILE``) > defaults

Every field is validated by pydantic. The root :class:`InferenceConfig`
contains one section per subsystem (general, exporter, package, validation).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field

from shared.config import apply_case_insensitive, deep_merge, parse_env
from shared.utils import yaml_safe

from .exceptions import InferenceConfigurationError

ENV_PREFIX = "INF_"


class GeneralConfig(BaseModel):
    """Package identity / device settings."""

    model_config = ConfigDict(extra="forbid")

    #: Name of the inference package (used in the manifest + README).
    package_name: str = "cropfusion"
    #: Semantic version of the inference package (``MAJOR.MINOR.PATCH``).
    version: str = "1.0.0"
    #: Device the exported model targets (``auto`` | ``cpu`` | ``cuda``).
    device: str = "auto"
    #: Output directory for the built package.
    output_dir: str = "artifacts/inference"
    #: Optional model version override (semver string) for ``model_version.json``.
    model_version: str | None = None
    #: Optional dataset version override (semver string) for
    #: ``dataset_version.json`` (defaults to the dataset manager version).
    dataset_version: str | None = None


class ExporterConfig(BaseModel):
    """Model export format settings."""

    model_config = ConfigDict(extra="forbid")

    #: Formats to produce: pytorch | torchscript | onnx.
    formats: list[str] = Field(
        default_factory=lambda: ["pytorch", "torchscript", "onnx"]
    )
    onnx_opset: int = Field(default=17, ge=9, le=22)
    torchscript_mode: str = Field(default="trace", pattern="^(trace|script)$")
    #: Batch size used for example inputs during export.
    export_batch_size: int = Field(default=2, ge=1)


class PackageConfig(BaseModel):
    """Inference-package artifact settings."""

    model_config = ConfigDict(extra="forbid")

    #: Author / maintainer shown in the generated README and LICENSE.
    author: str = "CropFusion"
    #: SPDX license identifier used in ``LICENSE``.
    license: str = "Apache-2.0"
    #: Human-readable description used in the README / manifest.
    description: str = (
        "CropFusion multimodal crop recommendation + yield prediction model "
        "(predict-only inference package)."
    )
    #: Whether to write a ``README.md`` and ``LICENSE``.
    docs: bool = True


class ValidationConfig(BaseModel):
    """Package validation settings."""

    model_config = ConfigDict(extra="forbid")

    #: Verify every artifact against ``checksums.json`` after building.
    verify_checksums: bool = True
    #: Run a smoke prediction through the packaged PyTorch model.
    smoke_test: bool = True
    #: Fail the build when any required artifact is missing.
    strict: bool = True


class InferenceConfig(BaseModel):
    """Root inference configuration."""

    model_config = ConfigDict(extra="forbid")

    name: str = "cropfusion_inference"
    general: GeneralConfig = Field(default_factory=GeneralConfig)
    exporter: ExporterConfig = Field(default_factory=ExporterConfig)
    package: PackageConfig = Field(default_factory=PackageConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)

    def to_yaml(self) -> str:
        return yaml.safe_dump(yaml_safe(self.model_dump()), sort_keys=False)

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.write_text(self.to_yaml(), encoding="utf-8")
        return out


def load_inference_config(
    config_path: str | Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> InferenceConfig:
    """Load and validate inference settings (env > YAML > defaults)."""
    env_map = dict(os.environ if env is None else env)
    if config_path is None:
        env_config = env_map.get("INF_CONFIG_FILE")
        config_path = env_config or None

    data: dict[str, Any] = {}
    if config_path is not None:
        config_file = Path(config_path)
        if not config_file.exists():
            raise InferenceConfigurationError(
                f"Inference config file not found: {config_file}",
                detail=str(config_file),
            )
        try:
            raw = yaml.safe_load(config_file.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise InferenceConfigurationError(
                f"Malformed inference YAML: {exc}", detail=str(config_file)
            ) from exc
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise InferenceConfigurationError(
                "Inference config root must be a mapping"
            )
        data = raw

    parsed_env = parse_env(env_map, prefix=ENV_PREFIX)
    parsed_env.pop("config_file", None)
    merged = deep_merge(data, parsed_env)
    merged = apply_case_insensitive(merged, InferenceConfig)
    try:
        return InferenceConfig.model_validate(merged)
    except Exception as exc:  # pydantic.ValidationError
        raise InferenceConfigurationError(
            f"Invalid inference configuration: {exc}"
        ) from exc


def save_inference_template(path: str | Path) -> Path:
    """Write an annotated YAML template of the default configuration."""
    out = Path(path)
    out.write_text(InferenceConfig().to_yaml(), encoding="utf-8")
    return out
