"""Inference engine port (architecture contract).

The engine runs the actual model forward pass. In R1.4 this is a port only:
the implementation will bind the exported inference package (``application/
models/``) to the trained CropFusion model in a later phase.

Contract notes:

- ``predict`` takes a location-only :class:`PredictionRequest`; season / year /
  context are resolved upstream and passed in as :class:`PredictionContext`.
- The engine must never bypass the exported model: no STAM or preprocessing
  code is embedded here. It consumes pre-built artifacts (feature scalers,
  label encoder, model weights) from the inference package.
- Implementations should be async (the API is async) and must expose
  ``status`` for health / readiness probes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..models import PredictionContext, PredictionRequest, PredictionResult


class InferenceEngine(ABC):
    """Port for the model-serving engine."""

    @abstractmethod
    async def predict(
        self,
        request: PredictionRequest,
        context: PredictionContext | None = None,
    ) -> PredictionResult:
        """Run one prediction and return a :class:`PredictionResult`."""

    @abstractmethod
    def status(self) -> dict[str, Any]:
        """Return readiness / model-version / device info for probes."""


__all__ = ["InferenceEngine"]
