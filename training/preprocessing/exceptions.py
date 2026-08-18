"""Exception hierarchy for the preprocessing pipeline.

Every failure raises :class:`PreprocessingError` (or a subclass) with a stable
machine-readable ``code`` (``PP-<AREA>-<NNN>``).
"""

from __future__ import annotations

from typing import Any

from shared.exceptions import CropFusionError


class PreprocessingError(CropFusionError):
    """Base class for all preprocessing errors."""

    code: str = "PP-ERROR"


class ConfigurationError(PreprocessingError):
    """Raised when the preprocessing configuration is invalid."""

    code = "PP-CONFIG-001"


class SampleRejectedError(PreprocessingError):
    """Raised when an observation fails the quality filter."""

    code = "PP-Q-001"


class MissingDependencyError(PreprocessingError):
    """Raised when an optional dependency (e.g. torch) is unavailable."""

    code = "PP-DEP-001"


class FitError(PreprocessingError):
    """Raised when a pipeline is used before it has been fitted."""

    code = "PP-FIT-001"


class ArtifactError(PreprocessingError):
    """Raised when a fitted artifact cannot be persisted or loaded."""

    code = "PP-ART-001"


class ShapeMismatchError(PreprocessingError):
    """Raised when tensors do not match the expected pipeline shapes."""

    code = "PP-SHAPE-001"


class DataContractViolationError(PreprocessingError):
    """Raised when a training corpus violates the training-data contract.

    R5.2.1 Task D: a corpus that mixes yield units (e.g. kg/ha village yields
    with a normalized district NPP proxy) or that would train a crop classifier
    on unlabeled ``-1`` observations is rejected up front instead of silently
    distorting the regression / classification targets.
    """

    code = "PP-DATA-CONTRACT-001"
