"""Base exception hierarchy for the backend.

Domain modules raise typed subclasses; a global handler maps them to
structured JSON responses with a stable machine-readable ``code``
(``B-<MODULE>-<NNN>``).
"""

from __future__ import annotations

from typing import Any


class BackendError(Exception):
    """Base class for all backend errors."""

    code: str = "B-ERROR"
    status_code: int = 500
    message: str

    def __init__(self, message: str, *, detail: Any = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail

    def __str__(self) -> str:
        text = f"{self.code}: {self.message}"
        if self.detail is not None:
            text += f" (detail={self.detail!r})"
        return text

    def to_response(self) -> dict[str, Any]:
        return {
            "error": {"code": self.code, "message": self.message, "detail": self.detail},
        }


class ConfigurationError(BackendError):
    code = "B-CONFIG-001"
    status_code = 500


class AuthenticationError(BackendError):
    code = "B-AUTH-001"
    status_code = 401


class AuthorizationError(BackendError):
    code = "B-AUTH-002"
    status_code = 403


class TokenError(BackendError):
    code = "B-AUTH-003"
    status_code = 401


class ValidationError(BackendError):
    code = "B-VALID-001"
    status_code = 422


class NotFoundError(BackendError):
    code = "B-NOTFOUND-001"
    status_code = 404


class ConflictError(BackendError):
    code = "B-CONFLICT-001"
    status_code = 409


class PredictionError(BackendError):
    code = "B-PREDICT-001"
    status_code = 500


class DatasetError(BackendError):
    code = "B-DATASET-001"
    status_code = 500


class GISError(BackendError):
    code = "B-GIS-001"
    status_code = 400


class InferenceError(BackendError):
    code = "B-INFER-001"
    status_code = 503


class ExplainabilityError(BackendError):
    code = "B-EXPLAIN-001"
    status_code = 500


class RateLimitError(BackendError):
    code = "B-RATE-001"
    status_code = 429


class ServiceUnavailableError(BackendError):
    code = "B-SVC-001"
    status_code = 503
