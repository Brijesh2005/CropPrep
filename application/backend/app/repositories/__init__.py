"""Repositories (data-access layer, repository pattern).

Repositories are request-scoped and constructed with an :class:`AsyncSession`;
they encapsulate all ORM access so services never touch SQLAlchemy directly.
"""

from __future__ import annotations

from app.repositories.prediction import ExplanationRepository, PredictionRepository
from app.repositories.user import UserRepository

__all__ = ["UserRepository", "PredictionRepository", "ExplanationRepository"]
