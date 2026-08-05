"""Inference services port (architecture contract).

``PredictionService`` is the orchestration layer that composes the GIS context
resolution, the inference engine, caching and history persistence for a single
``POST /predict`` call. It is the seam a future phase implements and the API
module calls — keeping the API layer free of pipeline details.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import PredictionRequest, PredictionResult


class PredictionService(ABC):
    """Port for the end-to-end prediction orchestration."""

    @abstractmethod
    async def predict(self, request: PredictionRequest) -> PredictionResult:
        """Resolve context, run inference and persist the result."""


__all__ = ["PredictionService"]
