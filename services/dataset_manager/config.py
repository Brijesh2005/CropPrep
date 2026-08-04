"""Configuration for the Dataset Manager.

Settings are resolved with the following precedence (highest wins):

1. Environment variables (prefix ``DM_``, nesting separated by ``__``).
2. YAML configuration file (``--config`` / ``DM_CONFIG_FILE``).
3. Sensible built-in defaults.

Nested settings are modelled as pydantic models so the whole tree is
validated at load time — unknown keys raise
:class:`InvalidConfigurationError`, wrong types are coerced, and enum /
literal fields reject out-of-range values.

Environment variable examples::

    DM_DATASET_ROOT=/data/cropfusion/datasets
    DM_DOWNLOAD__FORCE_DOWNLOAD=true
    DM_SCAN__WORKERS=16
    DM_LOG__LEVEL=DEBUG
    DM_VALIDATE__EXPECTED_YEARS="[2018, 2025]"

Note that list-valued options (e.g. ``expected_index_types``) are easiest to
configure via YAML; the environment can supply them as JSON arrays.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .exceptions import InvalidConfigurationError

#: Default Kaggle dataset handle for the primary image dataset.
DEFAULT_KAGGLE_HANDLE = (
    "shathanandabhatn/crop-yield-forecasting-karnataka-dakshina-kannada"
)

#: Environment variable prefix for the Dataset Manager.
ENV_PREFIX = "DM_"


class DownloadConfig(BaseModel):
    """Settings for :mod:`services.dataset_manager.downloader`."""

    model_config = ConfigDict(extra="forbid")

    kaggle_handle: str = DEFAULT_KAGGLE_HANDLE
    force_download: bool = False
    materialize: bool = True
    verify_integrity: bool = True
    link_method: str = "hardlink"  # "hardlink" | "copy"
    timeout_seconds: int = 0  # 0 => use the library default


class ScanConfig(BaseModel):
    """Settings for :mod:`services.dataset_manager.scanner`."""

    model_config = ConfigDict(extra="forbid")

    workers: int = Field(default=8, ge=1)
    hash_files: bool = False
    use_cache: bool = True
    follow_symlinks: bool = False


class ValidateConfig(BaseModel):
    """Settings for :mod:`services.dataset_manager.validator`."""

    model_config = ConfigDict(extra="forbid")

    fail_on_warning: bool = False
    expected_years: tuple[int, int] = (2018, 2025)
    expected_index_types: list[str] = ["NDVI", "EVI"]
    expected_resolutions: list[str] = ["R10m", "R20m"]
    require_metadata: bool = True


class CacheConfig(BaseModel):
    """Settings for :mod:`services.dataset_manager.cache_manager`."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    default_ttl_seconds: int = 86400
    max_entries: int = 10_000


class MetadataConfig(BaseModel):
    """Settings for :mod:`services.dataset_manager.metadata`."""

    model_config = ConfigDict(extra="forbid")

    store_type: str = "sqlite"  # "sqlite" | "parquet"
    db_path: Path | None = None  # resolved from dataset_root when None
    parquet_path: Path | None = None
    #: Compute SHA-256 content hashes during metadata generation.
    compute_hashes: bool = True
    #: Thread count used while generating metadata records in parallel.
    workers: int = Field(default=8, ge=1)


class RegistryConfig(BaseModel):
    """Settings for :mod:`services.dataset_manager.dataset_registry`."""

    model_config = ConfigDict(extra="forbid")

    db_path: Path | None = None
    auto_register_on_validate: bool = True


class LogConfig(BaseModel):
    """Settings for :mod:`services.dataset_manager.logger`."""

    model_config = ConfigDict(extra="forbid")

    level: str = "INFO"
    dir: Path | None = None  # None => console only
    max_bytes: int = 10 * 1024 * 1024
    backup_count: int = 5
    json_format: bool = True
    console: bool = True


class Settings(BaseModel):
    """Root settings object validated by pydantic."""

    model_config = ConfigDict(extra="forbid")

    #: Root of all managed datasets (raw/processed/metadata/cache below it).
    dataset_root: Path = Field(default=Path("datasets"))
    #: Directory name holding the canonical (materialised) copy of downloads.
    raw_dir_name: str = "raw"
    #: Directory name for derived / processed datasets.
    processed_dir_name: str = "processed"
    #: Directory name for internal state (sqlite stores, scan cache).
    state_dir_name: str = ".cropfusion"
    #: Default catalog (dataset) name used for registry + raw layout.
    catalog_name: str = "kaggle-crop-yield"
    #: Optional directory holding administrative boundary files (shapefiles /
    #: GeoJSON) that the Dataset Manager is allowed to serve. When None, only
    #: files under ``dataset_root`` may be loaded as geometries.
    admin_dir: Path | None = Field(default=None)

    download: DownloadConfig = Field(default_factory=DownloadConfig)
    scan: ScanConfig = Field(default_factory=ScanConfig)
    # ``validation`` (not ``validate``) to avoid shadowing pydantic's method.
    validation: ValidateConfig = Field(default_factory=ValidateConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    metadata: MetadataConfig = Field(default_factory=MetadataConfig)
    registry: RegistryConfig = Field(default_factory=RegistryConfig)
    logging: LogConfig = Field(default_factory=LogConfig)

    # -- Derived paths ------------------------------------------------------- #

    @property
    def raw_root(self) -> Path:
        return self.dataset_root / self.raw_dir_name

    @property
    def processed_root(self) -> Path:
        return self.dataset_root / self.processed_dir_name

    @property
    def state_root(self) -> Path:
        return self.dataset_root / self.state_dir_name

    @property
    def catalog_root(self) -> Path:
        """Canonical root of the primary catalog inside the raw directory."""
        return self.raw_root / self.catalog_name

    def metadata_db_path(self) -> Path:
        return self.metadata.db_path or (self.state_root / "metadata.db")

    def registry_db_path(self) -> Path:
        return self.registry.db_path or (self.state_root / "registry.db")

    def cache_db_path(self) -> Path:
        return self.state_root / "cache.db"

    def cache_dir(self) -> Path:
        return self.state_root / "cache"


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


def _normalise_key(key: str) -> str:
    """Lower-case + strip a field path segment for case-insensitive matching."""
    return key.lower().replace("-", "_")


def _parse_env(env: Mapping[str, str], prefix: str = ENV_PREFIX) -> dict[str, Any]:
    """Convert ``DM_<SECTION>__<FIELD>`` env vars into a nested dict.

    Values that look like JSON (``[...]``, ``{...}``, ``true``/``false``,
    integers) are parsed; everything else stays a string.
    """
    overrides: dict[str, Any] = {}
    for raw_key, raw_value in env.items():
        if not raw_key.startswith(prefix):
            continue
        path = raw_key[len(prefix):].split("__")
        path = [_normalise_key(part) for part in path if part]
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


def _apply_case_insensitive(data: dict[str, Any], schema: type[BaseModel]) -> dict[str, Any]:
    """Match config keys to pydantic field names case-insensitively."""
    field_names = {_normalise_key(name): name for name in schema.model_fields}
    return {field_names.get(_normalise_key(key), key): value for key, value in data.items()}


def load_settings(
    config_path: str | Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> Settings:
    """Load and validate the Dataset Manager settings.

    Args:
        config_path: Optional YAML file. When None, falls back to
            ``$DM_CONFIG_FILE``.
        env: Environment mapping; defaults to ``os.environ``.

    Returns:
        Validated :class:`Settings`.

    Raises:
        InvalidConfigurationError: When the YAML is malformed or settings
            fail pydantic validation.
    """
    env_map = dict(os.environ if env is None else env)

    # Locate the config file.
    if config_path is None:
        env_config = env_map.get("DM_CONFIG_FILE")
        config_path = env_config if env_config else None

    data: dict[str, Any] = {}
    if config_path is not None:
        config_file = Path(config_path)
        if not config_file.exists():
            raise InvalidConfigurationError(
                f"Configuration file not found: {config_file}", detail=str(config_file)
            )
        try:
            raw = yaml.safe_load(config_file.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise InvalidConfigurationError(
                f"Malformed YAML configuration: {exc}", detail=str(config_file)
            ) from exc
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise InvalidConfigurationError(
                "Configuration root must be a mapping", detail=str(config_file)
            )
        data = raw

    env_overrides = _parse_env(env_map)
    merged = deep_merge(data, env_overrides)
    merged = _apply_case_insensitive(merged, Settings)

    try:
        return Settings.model_validate(merged)
    except Exception as exc:  # pydantic.ValidationError
        raise InvalidConfigurationError(
            f"Invalid configuration: {exc}", detail=merged
        ) from exc


def save_settings_template(path: str | Path) -> Path:
    """Write an annotated YAML template of the current defaults.

    Useful to bootstrap a project configuration file.
    """
    template = {
        "dataset_root": "datasets",
        "raw_dir_name": "raw",
        "processed_dir_name": "processed",
        "state_dir_name": ".cropfusion",
        "catalog_name": "kaggle-crop-yield",
        "download": {
            "kaggle_handle": DEFAULT_KAGGLE_HANDLE,
            "force_download": False,
            "materialize": True,
            "verify_integrity": True,
            "link_method": "hardlink",
        },
        "scan": {"workers": 8, "hash_files": False, "use_cache": True},
        "validation": {
            "fail_on_warning": False,
            "expected_years": [2018, 2025],
            "expected_index_types": ["NDVI", "EVI"],
            "expected_resolutions": ["R10m", "R20m"],
            "require_metadata": True,
        },
        "cache": {"enabled": True, "default_ttl_seconds": 86400, "max_entries": 10000},
        "metadata": {"store_type": "sqlite"},
        "registry": {"auto_register_on_validate": True},
        "logging": {"level": "INFO", "json_format": True},
    }
    out = Path(path)
    out.write_text(yaml.safe_dump(template, sort_keys=False), encoding="utf-8")
    return out
