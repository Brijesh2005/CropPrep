"""Prediction history storage architecture (R1.4).

Defines the port a future phase binds to the existing enterprise schema:

- ``predictions``            — one row per prediction (crop, probs, yield,
                               confidence, model version, location admin fields)
- ``prediction_metadata``    — inputs / feature snapshot / weather / soil / tags
- ``explanations``           — per-prediction explanation artifacts

R1.4 makes NO schema changes; it only fixes the storage contract so the
inference services layer persists results without knowing the SQL.
"""

from __future__ import annotations

from .models import HistoryFilters, HistoryPage, HistoryRecord
from .store import PredictionHistoryStore

__version__ = "0.1.0"

__all__ = [
    "HistoryFilters",
    "HistoryPage",
    "HistoryRecord",
    "PredictionHistoryStore",
    "__version__",
]
