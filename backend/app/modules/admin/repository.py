"""Admin module repository."""

from __future__ import annotations

from app.repositories.prediction import PredictionRepository
from app.repositories.user import UserRepository

__all__ = ["PredictionRepository", "UserRepository"]
