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

import os
from pathlib import Path
from typing import Any, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field

from shared.config import apply_case_insensitive, deep_merge, parse_env

from .exceptions import InvalidConfigurationError

#: Default Kaggle dataset handle for the primary image dataset.
DEFAULT_KAGGLE_HANDLE = (
    "shathanandabhatn/crop-yield-forecasting-karnataka-dakshina-kannada"
)

#: Environment variable prefix for the Dataset Manager.
ENV_PREFIX = "DM_"


class DownloadConfig(BaseModel):
    """Settings for :mod:`training.dataset_manager.downloader`."""

    model_config = ConfigDict(extra="forbid")

    kaggle_handle: str = DEFAULT_KAGGLE_HANDLE
    force_download: bool = False
    materialize: bool = True
    verify_integrity: bool = True
    link_method: str = "hardlink"  # "hardlink" | "copy"
    timeout_seconds: int = 0  # 0 => use the library default


class ScanConfig(BaseModel):
    """Settings for :mod:`training.dataset_manager.scanner`."""

    model_config = ConfigDict(extra="forbid")

    workers: int = Field(default=8, ge=1)
    hash_files: bool = False
    use_cache: bool = True
    follow_symlinks: bool = False


class ValidateConfig(BaseModel):
    """Settings for :mod:`training.dataset_manager.validator`."""

    model_config = ConfigDict(extra="forbid")

    fail_on_warning: bool = False
    expected_years: tuple[int, int] = (2018, 2025)
    expected_index_types: list[str] = ["NDVI", "EVI"]
    expected_resolutions: list[str] = ["R10m", "R20m"]
    require_metadata: bool = True


class CacheConfig(BaseModel):
    """Settings for :mod:`training.dataset_manager.cache_manager`."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    default_ttl_seconds: int = 86400
    max_entries: int = 10_000


class MetadataConfig(BaseModel):
    """Settings for :mod:`training.dataset_manager.metadata`."""

    model_config = ConfigDict(extra="forbid")

    store_type: str = "sqlite"  # "sqlite" | "parquet"
    db_path: Path | None = None  # resolved from dataset_root when None
    parquet_path: Path | None = None
    #: Compute SHA-256 content hashes during metadata generation.
    compute_hashes: bool = True
    #: Thread count used while generating metadata records in parallel.
    workers: int = Field(default=8, ge=1)


class RegistryConfig(BaseModel):
    """Settings for :mod:`training.dataset_manager.dataset_registry`."""

    model_config = ConfigDict(extra="forbid")

    db_path: Path | None = None
    auto_register_on_validate: bool = True


class LogConfig(BaseModel):
    """Settings for :mod:`training.dataset_manager.logger`."""

    model_config = ConfigDict(extra="forbid")

    level: str = "INFO"
    dir: Path | None = None  # None => console only
    max_bytes: int = 10 * 1024 * 1024
    backup_count: int = 5
    json_format: bool = True
    console: bool = True


class TabularProviderConfig(BaseModel):
    """Settings for the Git-versioned tabular provider."""

    model_config = ConfigDict(extra="forbid")

    #: Root of the Git-versioned tabular CSVs. When None it resolves to
    #: ``<dataset_root>/tabular`` (e.g. ``training/datasets/tabular``).
    root: Path | None = None
    #: Glob patterns used for automatic discovery.
    patterns: list[str] = ["*.csv"]
    #: Default chunk size for streaming loads.
    chunk_size: int = 100_000
    #: Column suffix disambiguation for joins.
    join_suffixes: tuple[str, str] = ("_left", "_right")


class ImageProviderConfig(BaseModel):
    """Settings for the Kaggle imagery provider."""

    model_config = ConfigDict(extra="forbid")

    #: Kaggle dataset handle (defaults to the downloader's handle).
    handle: str | None = None
    #: Catalog directory name inside ``<dataset_root>/raw``.
    catalog_name: str | None = None
    materialize: bool = True
    verify_integrity: bool = True
    link_method: str = "hardlink"  # "hardlink" | "copy"
    force_download: bool = False


class ProviderEntryConfig(BaseModel):
    """One named provider registered in the provider registry.

    Enables multi-provider setups, priority ordering and future plugins: a
    provider can be disabled, re-ordered, or added entirely from configuration
    without touching code.

    Attributes:
        name: Registration name (e.g. ``git_repository_tabular``).
        kind: Provider kind (``tabular`` / ``image`` / ...).
        enabled: False disables the provider without removing it.
        priority: Higher values resolve first among providers of a kind.
        config: Free-form options forwarded to the provider factory.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    kind: str = "generic"
    enabled: bool = True
    priority: int = 100
    config: dict[str, Any] = Field(default_factory=dict)


class ProviderRegistryConfig(BaseModel):
    """Provider registry settings (registered providers + defaults).

    When the list is empty the manager registers its default tabular and image
    providers. Entries may override the defaults by matching the provider
    name, or register additional providers (future plugins).
    """

    model_config = ConfigDict(extra="forbid")

    providers: list[ProviderEntryConfig] = Field(default_factory=list)


class ProviderConfig(BaseModel):
    """Provider-layer settings (tab + image data sources)."""

    model_config = ConfigDict(extra="forbid")

    tabular: TabularProviderConfig = Field(default_factory=TabularProviderConfig)
    image: ImageProviderConfig = Field(default_factory=ImageProviderConfig)
    registry: ProviderRegistryConfig = Field(default_factory=ProviderRegistryConfig)


class ExecutionConfig(BaseModel):
    """Runtime execution settings (threading + streaming)."""

    model_config = ConfigDict(extra="forbid")

    #: Default thread-pool size used by parallel helpers.
    thread_pool_size: int = Field(default=8, ge=1)
    #: Default chunk size (rows) for streaming tabular loads.
    default_chunk_size: int = Field(default=100_000, ge=1)


class RasterCacheConfig(BaseModel):
    """Raster / patch cache settings (bounded in-memory + optional on-disk)."""

    model_config = ConfigDict(extra="forbid")

    #: Whether patch/window reads are cached at the manager layer.
    enabled: bool = True
    #: Maximum cached patches (LRU eviction).
    max_entries: int = Field(default=256, ge=1)
    #: TTL for cached raster reads.
    ttl_seconds: int = 3600
    #: On-disk cache directory; None resolves to ``<state_root>/cache/rasters``.
    cache_dir: Path | None = None

    def resolved_dir(self, state_root: Path) -> Path:
        return self.cache_dir or (state_root / "cache" / "rasters")


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
    providers: ProviderConfig = Field(default_factory=ProviderConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    raster: RasterCacheConfig = Field(default_factory=RasterCacheConfig)

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

    @property
    def tabular_root(self) -> Path:
        """Root of the Git-versioned tabular CSVs."""
        root = self.providers.tabular.root
        if root is not None:
            return Path(root)
        return self.dataset_root / "tabular"

    def metadata_db_path(self) -> Path:
        return self.metadata.db_path or (self.state_root / "metadata.db")

    def registry_db_path(self) -> Path:
        return self.registry.db_path or (self.state_root / "registry.db")

    def cache_db_path(self) -> Path:
        return self.state_root / "cache.db"

    def cache_dir(self) -> Path:
        return self.state_root / "cache"


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

    env_overrides = parse_env(env_map, ENV_PREFIX)
    merged = deep_merge(data, env_overrides)
    merged = apply_case_insensitive(merged, Settings)

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
        "providers": {
            "tabular": {
                "patterns": ["*.csv"],
                "chunk_size": 100000,
                "join_suffixes": ["_left", "_right"],
            },
            "image": {
                "handle": DEFAULT_KAGGLE_HANDLE,
                "materialize": True,
                "verify_integrity": True,
                "link_method": "hardlink",
                "force_download": False,
            },
            "registry": {
                "providers": [
                    {
                        "name": "git_repository_tabular",
                        "kind": "tabular",
                        "enabled": True,
                        "priority": 100,
                    },
                    {
                        "name": "kaggle_hub_image",
                        "kind": "image",
                        "enabled": True,
                        "priority": 100,
                    },
                ]
            },
        },
        "execution": {"thread_pool_size": 8, "default_chunk_size": 100000},
        "raster": {
            "enabled": True,
            "max_entries": 256,
            "ttl_seconds": 3600,
            "cache_dir": None,
        },
    }
    out = Path(path)
    out.write_text(yaml.safe_dump(template, sort_keys=False), encoding="utf-8")
    return out
