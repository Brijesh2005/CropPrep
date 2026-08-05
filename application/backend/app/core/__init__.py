"""Application core: configuration, logging, security, DI, database."""

from __future__ import annotations

from app.core.config import Settings, load_settings, save_settings_template
from app.core.container import Container
from app.core.database import Base, Database
from app.core.exceptions import (
    AuthenticationError,
    AuthorizationError,
    BackendError,
    ConflictError,
    ConfigurationError,
    DatasetError,
    ExplainabilityError,
    GISError,
    InferenceError,
    NotFoundError,
    PredictionError,
    RateLimitError,
    ServiceUnavailableError,
    TokenError,
    ValidationError,
)
from app.core.security import (
    RBAC,
    ROLE_ADMIN,
    ROLE_ANALYST,
    ROLE_USER,
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)

__all__ = [
    "Settings",
    "load_settings",
    "save_settings_template",
    "Container",
    "Base",
    "Database",
    "BackendError",
    "ConfigurationError",
    "AuthenticationError",
    "AuthorizationError",
    "TokenError",
    "ValidationError",
    "NotFoundError",
    "ConflictError",
    "PredictionError",
    "DatasetError",
    "GISError",
    "InferenceError",
    "ExplainabilityError",
    "RateLimitError",
    "ServiceUnavailableError",
    "RBAC",
    "ROLE_ADMIN",
    "ROLE_ANALYST",
    "ROLE_USER",
    "create_access_token",
    "create_refresh_token",
    "decode_token",
    "hash_password",
    "verify_password",
]
