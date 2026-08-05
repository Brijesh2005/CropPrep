"""Backend configuration (pydantic v2).

Settings resolve env (``BACKEND_<SECTION>__<KEY>``) > YAML (``BACKEND_CONFIG_FILE``)
> defaults, mirroring the other CropFusion packages. Every field is validated.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field

from training.dataset_manager.config import _apply_case_insensitive, _parse_env, deep_merge

ENV_PREFIX = "BACKEND_"


class DatabaseSettings(BaseModel):
    """Async SQLAlchemy settings."""

    model_config = ConfigDict(extra="forbid")

    #: ``sqlite+aiosqlite`` (tests / dev) or ``postgresql+asyncpg`` (prod).
    url: str = "sqlite+aiosqlite:///./cropfusion.db"
    echo: bool = False
    #: Optional PostGIS extension / spatial settings for PostgreSQL.
    postgis: bool = False
    #: Connection-pool sizing (PostgreSQL only).
    pool_size: int = 10
    max_overflow: int = 20
    pool_recycle_seconds: int = 1800


class PasswordPolicySettings(BaseModel):
    """Password complexity + account-lockout policy (Phase 10)."""

    model_config = ConfigDict(extra="forbid")

    min_length: int = 8
    max_length: int = 128
    require_uppercase: bool = False
    require_lowercase: bool = False
    require_digit: bool = False
    require_special: bool = False
    #: Reject passwords containing the user's email (case-insensitive).
    prevent_email_substring: bool = True
    #: Failed logins before the account is temporarily locked.
    max_failed_attempts: int = 5
    #: Lockout duration after too many failures.
    lockout_minutes: int = 15
    #: TTL for password-reset tokens.
    reset_token_ttl_minutes: int = 30
    #: TTL for email-verification tokens.
    email_verify_ttl_hours: int = 24


class SecuritySettings(BaseModel):
    """JWT / password / RBAC settings."""

    model_config = ConfigDict(extra="forbid")

    secret_key: str = "change-me-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    #: Bcrypt / PBKDF2 rounds used by passlib (legacy hashes).
    password_scheme: str = "pbkdf2_sha256"
    #: Argon2id parameters used by the Phase 10 password service.
    password_algorithm: str = "argon2"
    argon2_time_cost: int = 3
    argon2_memory_kib: int = 65536  # 64 MiB
    argon2_parallelism: int = 4
    #: Issuer claim stamped on tokens.
    issuer: str = "cropfusion-backend"
    #: Per-user session lifetime.
    session_ttl_hours: int = 72
    #: Set Secure/HttpOnly/SameSite on cookie-based auth (production).
    secure_cookies: bool = False
    #: Maximum active sessions per user.
    max_sessions_per_user: int = 10
    #: Password + lockout policy.
    password_policy: PasswordPolicySettings = Field(default_factory=PasswordPolicySettings)


class DatasetSettings(BaseModel):
    """Dataset Manager / STAM wiring."""

    model_config = ConfigDict(extra="forbid")

    dataset_root: str | None = None
    catalog_name: str = "kaggle-crop-yield"
    admin_dir: str | None = None
    #: STAM patch size.
    patch_size: int = 128
    #: STAM image resolution.
    image_resolution: str = "R10m"
    require_pairs: bool = True
    #: Validate datasets at startup.
    validate_on_startup: bool = True


class ModelSettings(BaseModel):
    """Trained-model loading settings."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    #: Path to a Phase 5/6 checkpoint (``.pt``).
    checkpoint_path: str | None = None
    #: Path to a model config YAML (falls back to the checkpoint's config).
    model_config_path: str | None = None
    #: Path to a fitted preprocessor directory (optional).
    preprocessor_dir: str | None = None
    device: str = "auto"  # auto | cpu | cuda
    batch_size: int = 1
    warmup: bool = True


class InferenceSettings(BaseModel):
    """Inference-engine settings."""

    model_config = ConfigDict(extra="forbid")

    queue_size: int = 100
    max_workers: int = 2
    timeout_seconds: float = 60.0
    cache_ttl_seconds: int = 3600
    enable_cache: bool = True
    #: Fall back to a heuristic prediction when the model is unavailable.
    enable_fallback: bool = True


class CacheSettings(BaseModel):
    """Prediction cache backend."""

    model_config = ConfigDict(extra="forbid")

    backend: str = "memory"  # memory | redis
    redis_url: str = "redis://localhost:6379/0"
    default_ttl_seconds: int = 3600


class RedisSettings(BaseModel):
    """Redis (Phase 10): sessions, spatial cache, reports, rate limiting."""

    model_config = ConfigDict(extra="forbid")

    url: str = "redis://localhost:6379/0"
    enabled: bool = False
    session_ttl_seconds: int = 3 * 24 * 3600
    prediction_ttl_seconds: int = 3600
    spatial_ttl_seconds: int = 3600
    report_ttl_seconds: int = 3600
    rate_limit_ttl_seconds: int = 60
    key_prefix: str = "cf10"


class RateLimitSettings(BaseModel):
    """Rate limiting settings."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    requests_per_minute: int = 60
    storage: str = "memory"  # memory | redis


class SeedSettings(BaseModel):
    """Bootstrap data seeding (Phase 10)."""

    model_config = ConfigDict(extra="forbid")

    on_startup: bool = False
    include_boundaries: bool = True
    #: Optional path to the ICRISAT district-level CSV (falls back to a small
    #: synthetic district set when unset).
    csv_path: str | None = None


class CORSSettings(BaseModel):
    """Cross-origin resource sharing."""

    model_config = ConfigDict(extra="forbid")

    allow_origins: list[str] = Field(default_factory=lambda: ["*"])
    allow_credentials: bool = False
    allow_methods: list[str] = Field(
        default_factory=lambda: ["*"]
    )
    allow_headers: list[str] = Field(default_factory=lambda: ["*"])


class MonitoringSettings(BaseModel):
    """Observability: Prometheus metrics + OpenTelemetry tracing (Phase 11)."""

    model_config = ConfigDict(extra="forbid")

    #: Expose request / inference / cache metrics on ``GET /metrics``.
    prometheus_enabled: bool = True
    #: Metric-name namespace (``cropfusion_requests_total``).
    prometheus_namespace: str = "cropfusion"
    #: Instrument the app with OpenTelemetry distributed tracing.
    tracing_enabled: bool = False
    #: ``console`` | ``otlp`` — where spans are exported.
    tracing_exporter: str = "console"
    #: OTLP endpoint used when ``tracing_exporter == "otlp"``.
    otlp_endpoint: str = "http://localhost:4317"


class LogSettings(BaseModel):
    """Structured logging settings."""

    model_config = ConfigDict(extra="forbid")

    level: str = "INFO"
    #: JSON output for machine parsing.
    json_logs: bool = True
    #: Optional rotating log file (``None`` = console only).
    log_dir: str | None = None
    max_bytes: int = 10 * 1024 * 1024
    backup_count: int = 5
    correlation_header: str = "X-Request-ID"


class Settings(BaseModel):
    """Root backend settings."""

    model_config = ConfigDict(extra="forbid")

    app_name: str = "CropFusion Backend"
    environment: str = "development"
    debug: bool = True
    api_prefix: str = "/api/v1"
    version: str = "0.1.0"
    #: UI/API mode: ``farmer`` (location-only prediction; year/season fields
    #: are removed and the season is auto-resolved from the date) or
    #: ``research`` (advanced year/season controls re-enabled).
    application_mode: str = Field(default="farmer", pattern="^(farmer|research)$")

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    dataset: DatasetSettings = Field(default_factory=DatasetSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    inference: InferenceSettings = Field(default_factory=InferenceSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    rate_limit: RateLimitSettings = Field(default_factory=RateLimitSettings)
    seed: SeedSettings = Field(default_factory=SeedSettings)
    cors: CORSSettings = Field(default_factory=CORSSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    log: LogSettings = Field(default_factory=LogSettings)

    def to_yaml(self) -> str:
        return yaml.safe_dump(_yaml_safe(self.model_dump()), sort_keys=False)

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.write_text(self.to_yaml(), encoding="utf-8")
        return out


def load_settings(
    config_path: str | Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> Settings:
    """Load and validate backend settings (env > YAML > defaults)."""
    env_map = dict(os.environ if env is None else env)
    if config_path is None:
        env_config = env_map.get("BACKEND_CONFIG_FILE")
        config_path = env_config or None

    data: dict[str, Any] = {}
    if config_path is not None:
        config_file = Path(config_path)
        if config_file.exists():
            raw = yaml.safe_load(config_file.read_text(encoding="utf-8"))
            data = raw if isinstance(raw, dict) else {}

    parsed_env = _parse_env(env_map, prefix=ENV_PREFIX)
    parsed_env.pop("config_file", None)
    merged = deep_merge(data, parsed_env)
    merged = _apply_case_insensitive(merged, Settings)
    return Settings.model_validate(merged)


def save_settings_template(path: str | Path) -> Path:
    """Write an annotated YAML template of the default settings."""
    out = Path(path)
    out.write_text(Settings().to_yaml(), encoding="utf-8")
    return out


def _yaml_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _yaml_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_yaml_safe(v) for v in value]
    return value
