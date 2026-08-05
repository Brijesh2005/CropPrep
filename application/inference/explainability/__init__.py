"""Inference explainability port (architecture contract).

Produces human-readable explanations for a :class:`PredictionResult` (which
features, which historical context, how confident) without invoking the heavy
Training Platform explainers. The training explainability algorithms remain
in the Training Platform; this port defines the lightweight serving-time
surface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..models import PredictionContext, PredictionResult


class PredictionExplainer(ABC):
    """Port for serving-time explanation generation."""

    @abstractmethod
    async def explain(
        self,
        result: PredictionResult,
        context: PredictionContext | None = None,
    ) -> dict[str, Any]:
        """Return an explanation summary dict (features / reasoning / confidence)."""


__all__ = ["PredictionExplainer"]
