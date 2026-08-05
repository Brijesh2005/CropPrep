"""Data drift monitoring framework for CropFusion.

Detects distribution shift between a trusted *reference* dataset (training /
launch baseline) and a *current* production dataset across five dimensions:

* feature drift  (:class:`~quality.drift.feature_drift.FeatureDriftAnalyzer`)
* label drift    (:class:`~quality.drift.label_drift.LabelDriftAnalyzer`)
* prediction drift (:class:`~quality.drift.prediction_drift.PredictionDriftAnalyzer`)
* spatial drift  (:class:`~quality.drift.spatial_drift.SpatialDriftAnalyzer`)
* temporal drift (:class:`~quality.drift.temporal_drift.TemporalDriftAnalyzer`)

Every analyzer returns a structured, serialisable result; a
:class:`~quality.drift.monitor.DriftMonitor` orchestrates them and emits a
single :class:`~quality.drift.result.DriftReport` rendered to JSON / CSV /
HTML / PDF via :mod:`training.quality.drift.report`.
"""

from __future__ import annotations

from .config import DriftConfig
from .exceptions import DriftError
from .monitor import DriftMonitor
from .report import ReportWriter

__all__ = [
    "DriftConfig",
    "DriftError",
    "DriftMonitor",
    "ReportWriter",
]
