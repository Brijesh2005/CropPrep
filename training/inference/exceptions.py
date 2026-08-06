"""Exception hierarchy for the inference package (Phase R5).

Every failure raises :class:`InferenceError` (or a subclass) with a stable
machine-readable ``code`` (``INF-<AREA>-<NNN>``), mirroring the convention used
by the other CropFusion packages (``TR-*``, ``EV-*``, ``MOD-*``).
"""

from __future__ import annotations

from shared.exceptions import CropFusionError


class InferenceError(CropFusionError):
    """Base class for all inference-package errors."""

    code: str = "INF-ERROR"


class InferenceConfigurationError(InferenceError):
    """Raised when the inference configuration is invalid or inconsistent."""

    code = "INF-CONFIG-001"


class ExportError(InferenceError):
    """Raised when a model export fails."""

    code = "INF-EXPORT-001"


class PackageBuildError(InferenceError):
    """Raised when the inference package cannot be generated."""

    code = "INF-PKG-001"


class PackageValidationError(InferenceError):
    """Raised when an inference package fails validation."""

    code = "INF-VAL-001"


class VersioningError(InferenceError):
    """Raised when artifact versioning fails."""

    code = "INF-VER-001"


class DatasetSourceError(InferenceError):
    """Raised when the dataset sources for a package cannot be produced."""

    code = "INF-DS-001"
