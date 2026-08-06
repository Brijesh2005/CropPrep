"""Imports every ORM model so ``Base.metadata`` is fully populated before
``Database.create_all()`` runs. NEW FILE — this package did not exist in R5."""

from app.models.prediction import ExplanationRecord, Prediction  # noqa: F401

__all__ = ["ExplanationRecord", "Prediction"]
