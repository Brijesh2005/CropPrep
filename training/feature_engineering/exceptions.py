"""Exception hierarchy for the feature-engineering package.

Every failure raises :class:`FeatureEngineeringError` (or a subclass) carrying
a stable machine-readable ``code`` (``FE-<AREA>-<NNN>``) so callers can catch
the base type and still discriminate between failure modes.
"""

from __future__ import annotations

from typing import Any

from shared.exceptions import CropFusionError


class FeatureEngineeringError(CropFusionError):
    """Base class for all feature-engineering errors."""

    code: str = "FE-ERROR"


class FeatureConfigError(FeatureEngineeringError):
    """Raised when feature-engineering configuration is invalid."""

    code = "FE-CONFIG-001"


class FeatureBuilderError(FeatureEngineeringError):
    """Raised when a feature builder cannot process an observation."""

    code = "FE-BUILD-001"


class MissingExtractorError(FeatureBuilderError):
    """Raised when image features require a patch extractor and none is given."""

    code = "FE-BUILD-002"


class FeatureFrameError(FeatureEngineeringError):
    """Raised when a feature frame cannot be assembled."""

    code = "FE-FRAME-001"
