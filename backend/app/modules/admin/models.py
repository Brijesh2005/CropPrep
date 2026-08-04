"""Admin module models (re-export)."""

from __future__ import annotations

from app.models.prediction import Prediction
from app.models.user import User

__all__ = ["Prediction", "User"]
