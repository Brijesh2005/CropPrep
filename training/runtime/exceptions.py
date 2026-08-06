"""Exception hierarchy for the release runtime (Phase R6).

Every failure raises :class:`RuntimeEnvironmentError` (or a subclass) with a
stable machine-readable ``code`` (``RT-<AREA>-<NNN>``), mirroring the
convention used by the other CropFusion packages (``TR-*``, ``EVAL-*``,
``INF-*``).
"""

from __future__ import annotations

from shared.exceptions import CropFusionError


class RuntimeEnvironmentError(CropFusionError):
    """Base class for all release-runtime errors."""

    code: str = "RT-ERROR"


class RuntimeConfigurationError(RuntimeEnvironmentError):
    """Raised when the runtime configuration is invalid or inconsistent."""

    code = "RT-CONFIG-001"


class ReleaseError(RuntimeEnvironmentError):
    """Base class for release discovery / management errors."""

    code = "RT-REL-001"


class ReleaseNotFoundError(ReleaseError):
    """Raised when a requested release version does not exist."""

    code = "RT-REL-002"


class ReleaseLayoutError(ReleaseError):
    """Raised when a release directory does not match the release layout."""

    code = "RT-REL-003"


class ReleaseValidationError(ReleaseError):
    """Raised when a release fails validation (strict mode)."""

    code = "RT-REL-004"


class ReleaseActivationError(ReleaseError):
    """Raised when a release cannot be activated."""

    code = "RT-ACT-001"


class ReleaseRollbackError(ReleaseError):
    """Raised when no previous release is available for rollback."""

    code = "RT-ROLLBACK-001"


class ReleasePackagingError(ReleaseError):
    """Raised when an inference package cannot be turned into a release."""

    code = "RT-PKG-001"


class ModelLoadError(RuntimeEnvironmentError):
    """Raised when the model cannot be loaded from a release."""

    code = "RT-MODEL-001"


class ModelWarmupError(RuntimeEnvironmentError):
    """Raised when model warm-up fails."""

    code = "RT-WARMUP-001"


class PreprocessLoadError(RuntimeEnvironmentError):
    """Raised when the preprocessing pipelines cannot be loaded."""

    code = "RT-PRE-001"


class MetadataLoadError(RuntimeEnvironmentError):
    """Raised when the release metadata cannot be loaded."""

    code = "RT-META-001"


class DependencyError(RuntimeEnvironmentError):
    """Raised when a required runtime dependency is missing."""

    code = "RT-DEP-001"


class MemoryLimitError(RuntimeEnvironmentError):
    """Raised when the runtime exceeds a configured memory limit."""

    code = "RT-MEM-001"


class HealthError(RuntimeEnvironmentError):
    """Raised when a health check cannot be produced."""

    code = "RT-HEALTH-001"
