"""Configuration for the release runtime (Phase R6).

Everything is configurable through YAML (or ``RT_*`` env vars), mirroring the
resolution order of the other CropFusion packages:

    env (``RT_<SECTION>__<KEY>``) > YAML (``RT_CONFIG_FILE``) > defaults

Every field is validated by pydantic. The root :class:`RuntimeConfig`
contains one section per subsystem (general, model, preprocess, metadata,
cache, memory, health, hot_reload, validation).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field

from shared.config import apply_case_insensitive, deep_merge, parse_env
from shared.utils import yaml_safe

from .exceptions import RuntimeConfigurationError

ENV_PREFIX = "RT_"


class GeneralConfig(BaseModel):
    """Runtime identity / device settings."""

    model_config = ConfigDict(extra="forbid")

    #: Name of the runtime instance (used in state + health reports).
    name: str = "cropfusion_runtime"
    #: Root directory under which releases are discovered / stored.
    releases_root: str = "releases"
    #: Device the runtime targets (``auto`` | ``cpu`` | ``cuda``).
    device: str = "auto"
    #: Seed for deterministic warm-up input generation.
    seed: int | None = None
    #: Log level for the runtime logger.
    log_level: str = "INFO"


class ModelLoadConfig(BaseModel):
    """Model loading / warm-up settings."""

    model_config = ConfigDict(extra="forbid")

    #: Backend used to serve the model: ``auto`` | ``pytorch`` | ``torchscript``
    #: | ``onnx``. ``auto`` prefers pytorch, then torchscript, then onnx.
    backend: str = "auto"
    #: Precision the loaded model is converted to (``float32`` | ``float16`` |
    #: ``bfloat16``). Only applies to the pytorch backend.
    precision: str = "float32"
    #: Device override for the pytorch backend (defaults to ``general.device``).
    device: str = "auto"
    #: Number of warm-up forward passes to run after loading.
    warmup_steps: int = Field(default=1, ge=0)
    #: Batch size used for the warm-up input batch.
    warmup_batch_size: int = Field(default=2, ge=1)
    #: ONNX execution providers (used by the onnx backend).
    onnx_providers: list[str] = Field(
        default_factory=lambda: ["CPUExecutionProvider"]
    )


class PreprocessConfig(BaseModel):
    """Preprocessing pipeline loading settings."""

    model_config = ConfigDict(extra="forbid")

    #: Whether the feature_scalers / label_encoder pipelines are required.
    required: bool = True


class MetadataConfig(BaseModel):
    """Release metadata loading settings."""

    model_config = ConfigDict(extra="forbid")

    #: Whether the metadata artefacts are required.
    required: bool = True
    #: Whether ``feature_lookup.parquet`` is required (the packager always
    #: generates it).
    feature_lookup_required: bool = True
    #: Approximate memory budget for cached dataframes, in megabytes.
    cache_size_mb: int = Field(default=64, ge=0)
    #: Maximum number of cached dataframe entries.
    cache_max_entries: int = Field(default=16, ge=0)


class CacheConfig(BaseModel):
    """Runtime cache settings (shared LRU used by the loaders)."""

    model_config = ConfigDict(extra="forbid")

    #: Whether caching is enabled at all.
    enabled: bool = True
    #: Hard memory budget for cached values, in bytes.
    max_bytes: int = Field(default=256 * 1024 * 1024, ge=0)
    #: Maximum number of cached entries.
    max_entries: int = Field(default=256, ge=0)
    #: Optional TTL in seconds after which entries are evicted.
    ttl_seconds: int | None = Field(default=None, ge=0)


class MemoryConfig(BaseModel):
    """Memory monitoring / limits."""

    model_config = ConfigDict(extra="forbid")

    #: Hard process-RSS limit in MB; loading refuses to continue above it.
    limit_mb: int | None = Field(default=None, ge=0)
    #: Soft process-RSS limit in MB; exceeding it evicts cache + warns.
    soft_limit_mb: int | None = Field(default=None, ge=0)
    #: How often the memory monitor is refreshed (seconds).
    check_interval_seconds: float = Field(default=5.0, gt=0)


class HealthConfig(BaseModel):
    """Health reporting settings."""

    model_config = ConfigDict(extra="forbid")

    #: Whether the runtime maintains health state.
    enabled: bool = True
    #: Suggested health-polling interval (seconds) for downstream consumers.
    interval_seconds: float = Field(default=10.0, gt=0)
    #: How long a release is allowed to take to reach readiness (seconds).
    readiness_timeout_seconds: float = Field(default=60.0, gt=0)


class HotReloadConfig(BaseModel):
    """Hot-reload settings."""

    model_config = ConfigDict(extra="forbid")

    #: Whether hot reload is enabled.
    enabled: bool = False
    #: Poll interval for the background hot-reload watcher (seconds).
    poll_interval_seconds: float = Field(default=30.0, gt=0)
    #: When a change is detected, reload the active release automatically.
    auto_reload: bool = True
    #: Optional cap on automatic reloads (``None`` = unlimited).
    max_reloads: int | None = Field(default=None, ge=0)


class ValidationConfig(BaseModel):
    """Release validation settings."""

    model_config = ConfigDict(extra="forbid")

    verify_checksums: bool = True
    verify_manifest: bool = True
    verify_versions: bool = True
    verify_dependencies: bool = True
    verify_compatibility: bool = True
    #: Run a forward-pass smoke test during release validation.
    smoke_test: bool = True
    #: Fail the operation (activate / start) when validation fails.
    strict: bool = True


class RuntimeConfig(BaseModel):
    """Root runtime configuration."""

    model_config = ConfigDict(extra="forbid")

    name: str = "cropfusion_runtime"
    general: GeneralConfig = Field(default_factory=GeneralConfig)
    model: ModelLoadConfig = Field(default_factory=ModelLoadConfig)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    metadata: MetadataConfig = Field(default_factory=MetadataConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    health: HealthConfig = Field(default_factory=HealthConfig)
    hot_reload: HotReloadConfig = Field(default_factory=HotReloadConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)

    def to_yaml(self) -> str:
        return yaml.safe_dump(yaml_safe(self.model_dump()), sort_keys=False)

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.write_text(self.to_yaml(), encoding="utf-8")
        return out


def load_runtime_config(
    config_path: str | Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> RuntimeConfig:
    """Load and validate runtime settings (env > YAML > defaults)."""
    env_map = dict(os.environ if env is None else env)
    if config_path is None:
        env_config = env_map.get("RT_CONFIG_FILE")
        config_path = env_config or None

    data: dict[str, Any] = {}
    if config_path is not None:
        config_file = Path(config_path)
        if not config_file.exists():
            raise RuntimeConfigurationError(
                f"Runtime config file not found: {config_file}",
                detail=str(config_file),
            )
        try:
            raw = yaml.safe_load(config_file.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise RuntimeConfigurationError(
                f"Malformed runtime YAML: {exc}", detail=str(config_file)
            ) from exc
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise RuntimeConfigurationError(
                "Runtime config root must be a mapping"
            )
        data = raw

    parsed_env = parse_env(env_map, prefix=ENV_PREFIX)
    parsed_env.pop("config_file", None)
    merged = deep_merge(data, parsed_env)
    merged = apply_case_insensitive(merged, RuntimeConfig)
    try:
        return RuntimeConfig.model_validate(merged)
    except Exception as exc:  # pydantic.ValidationError
        raise RuntimeConfigurationError(
            f"Invalid runtime configuration: {exc}"
        ) from exc


def save_runtime_template(path: str | Path) -> Path:
    """Write an annotated YAML template of the default configuration."""
    out = Path(path)
    out.write_text(RuntimeConfig().to_yaml(), encoding="utf-8")
    return out
