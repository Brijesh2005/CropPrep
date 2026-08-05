"""Shared exception hierarchy for the CropFusion platforms.

All platform and domain errors derive from :class:`CropFusionError` which
carries a stable machine-readable ``code``, a human readable ``message``,
optional structured ``detail`` and an optional ``suggested_resolution``.

Domain specific bases let platforms subclass with their own stable prefixes
(``DM-``, ``TD-``, ``ST-``, ``PPT-``, ``MOD-``, ``EXP-``, ``ML-``, ``API-``)
while still being catchable as their shared parent.
"""

from __future__ import annotations

from .base import CropFusionError
from .config import (
    ConfigurationError,
    InvalidEnvironmentError,
    MissingSettingError,
    SettingsTemplateError,
)
from .data import (
    CacheError,
    DataError,
    IntegrityError,
    NotFoundError,
    ProviderError,
    ScannerError,
    StorageError,
    UnsupportedFormatError,
)
from .io import FileAccessError, SerializationError, UnsupportedSerializerError
from .logging import LoggingConfigurationError
from .model import CheckpointError, InferenceError, ModelError, TrainingError
from .prediction import ModelNotLoadedError, PredictionError, PredictionInputError
from .security import AuthenticationError, AuthorizationError, SecurityError, TokenError
from .validation import ValidationFailedError, ValidationNotSupportedError
from .versioning import (
    InvalidVersionError,
    RegistryError,
    VersionError,
    VersionNotFoundError,
)

__all__ = [
    "AuthenticationError",
    "AuthorizationError",
    "CacheError",
    "CheckpointError",
    "ConfigurationError",
    "CropFusionError",
    "DataError",
    "FileAccessError",
    "InferenceError",
    "IntegrityError",
    "InvalidEnvironmentError",
    "InvalidVersionError",
    "LoggingConfigurationError",
    "MissingSettingError",
    "ModelError",
    "ModelNotLoadedError",
    "NotFoundError",
    "PredictionError",
    "PredictionInputError",
    "ProviderError",
    "RegistryError",
    "ScannerError",
    "SecurityError",
    "SerializationError",
    "SettingsTemplateError",
    "StorageError",
    "TokenError",
    "TrainingError",
    "UnsupportedFormatError",
    "UnsupportedSerializerError",
    "ValidationFailedError",
    "ValidationNotSupportedError",
    "VersionError",
    "VersionNotFoundError",
]
