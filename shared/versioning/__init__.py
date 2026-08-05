"""Versioning shared across the CropFusion platforms.

Provides :class:`SemanticVersion` (MAJOR.MINOR.PATCH parsing, comparison and
bumping), version metadata types for datasets / models / inference /
applications, and the :class:`VersionProvider` port.
"""

from __future__ import annotations

from .provider import VersionProvider
from .semver import SemanticVersion
from .versions import (
    ApplicationVersion,
    DatasetVersion,
    InferenceVersion,
    ModelVersion,
    VersionInfo,
)

__all__ = [
    "ApplicationVersion",
    "DatasetVersion",
    "InferenceVersion",
    "ModelVersion",
    "SemanticVersion",
    "VersionInfo",
    "VersionProvider",
]
