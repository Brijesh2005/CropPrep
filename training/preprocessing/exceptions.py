"""Exception hierarchy for the preprocessing pipeline.

Every failure raises :class:`PreprocessingError` (or a subclass) with a stable
machine-readable ``code`` (``PP-<AREA>-<NNN>``).
"""

from __future__ import annotations

from typing import Any


class PreprocessingError(Exception):
    """Base class for all preprocessing errors."""

    code: str = "PP-ERROR"

    def __init__(self, message: str, *, detail: Any = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail

    def __str__(self) -> str:
        text = f"{self.code}: {self.message}"
        if self.detail is not None:
            text += f" (detail={self.detail!r})"
        return text


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
