"""Explainability module exceptions."""

from __future__ import annotations

from app.core.exceptions import ExplainabilityError, NotFoundError

__all__ = ["ExplainabilityError", "NotFoundError"]


class ExplanationNotFoundError(NotFoundError):
    code = "B-EXPLAIN-100"
