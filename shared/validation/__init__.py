"""Validation framework shared across the CropFusion platforms.

Provides the :class:`Validator` port, :class:`ValidationResult` /
:class:`ValidationIssue` result types, concrete validators for CSV files,
images/rasters, metadata records, config mappings, schemas and versions, and
a :class:`ValidatorRegistry` for dispatch.
"""

from __future__ import annotations

from .base import Validator, ValidationIssue, ValidationResult
from .registry import ValidatorRegistry, default_registry
from .validators import (
    ConfigValidator,
    CsvValidator,
    ImageValidator,
    MetadataValidator,
    SchemaValidator,
    VersionValidator,
)

for _validator in (
    CsvValidator(),
    ImageValidator(),
    MetadataValidator(),
    ConfigValidator(),
    VersionValidator(),
):
    default_registry.register(_validator)


def validate(target: object, name: str, **context: object) -> ValidationResult:
    """Validate ``target`` using the validator registered as ``name``."""
    return default_registry.get(name).validate(target, **context)


__all__ = [
    "ConfigValidator",
    "CsvValidator",
    "ImageValidator",
    "MetadataValidator",
    "SchemaValidator",
    "ValidationIssue",
    "ValidationResult",
    "ValidationNotSupportedError",
    "Validator",
    "ValidatorRegistry",
    "VersionValidator",
    "default_registry",
    "validate",
]

from ..exceptions import ValidationNotSupportedError  # noqa: E402
