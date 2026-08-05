"""Inference platform package (Prediction Platform, R1.4).

Architecture skeleton for running the CropFusion model at serving time.
R1.4 prepares contracts only — there is intentionally no inference
implementation, no model loading and no backend API change.

Pipeline the skeleton describes:

    POST /predict (lon, lat)
        -> LocationResolver (application.gis)          # reverse geocoding
        -> SpatialResolver (application.gis)            # village/taluk/district
        -> HistoricalContextResolver (application.gis)  # season + climatology
        -> InferenceEngine (inference.engine)           # model forward (future)
        -> PredictionHistoryStore (application.history) # persist result

Sub-packages are pure ports (abstract base classes) that a later phase will
implement. They depend only on stdlib + ``shared`` — never on ``training``.
"""

from __future__ import annotations

from .models import PredictionContext, PredictionRequest, PredictionResult

__version__ = "0.1.0"

__all__ = [
    "PredictionContext",
    "PredictionRequest",
    "PredictionResult",
    "__version__",
]
