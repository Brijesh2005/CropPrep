"""Fairness evaluation framework for CropFusion.

Measures whether model performance and outcomes are consistent across
sensitive / operational groups — region, soil type, land size, farm size,
etc. Provides parity statistics (statistical parity, disparate impact,
equalized odds, equal opportunity, accuracy parity) plus calibration
assessment, and a regional/spatial view for geographic bias.
"""

from __future__ import annotations

from .config import FairnessConfig
from .evaluator import FairnessEvaluator
from .regional import RegionalFairnessEvaluator
from .report import FairnessReportWriter

__all__ = [
    "FairnessConfig",
    "FairnessEvaluator",
    "FairnessReportWriter",
    "RegionalFairnessEvaluator",
]
