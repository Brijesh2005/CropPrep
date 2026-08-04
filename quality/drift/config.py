"""Drift-metric configuration (validated thresholds)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DriftConfig(BaseModel):
    """Thresholds and hyper-parameters for every drift analyzer.

    Defaults follow industry-standard operating ranges:

    * PSI  < 0.10  low drift,  0.10–0.25 moderate,  > 0.25 significant.
    * JS   < 0.05  low,        0.05–0.10 moderate,  > 0.10 significant.
    * KS / chi² rejections use the classic p-value threshold of 0.05.
    """

    model_config = {"extra": "forbid"}

    #: Minimum samples for a distribution to be considered meaningful.
    min_samples: int = Field(default=30, ge=1)

    #: Number of bins used for numeric PSI / KL / JS histograms.
    bins: int = Field(default=10, ge=2)

    #: Probability floor applied before log ratios to avoid divide-by-zero.
    clip: float = Field(default=1e-4, gt=0.0)

    #: PSI severity boundaries (low, moderate).
    psi_thresholds: tuple[float, float] = (0.10, 0.25)

    #: JS divergence severity boundaries.
    js_thresholds: tuple[float, float] = (0.05, 0.10)

    #: Significance level for Kolmogorov–Smirnov tests.
    ks_alpha: float = Field(default=0.05, gt=0.0, le=1.0)

    #: Significance level for chi-squared tests.
    chi2_alpha: float = Field(default=0.05, gt=0.0, le=1.0)

    #: Significance level for label distribution tests.
    label_alpha: float = Field(default=0.05, gt=0.0, le=1.0)

    #: Fraction of spatial cells that may be new before coverage drift fires.
    spatial_max_novel_cells: float = Field(default=0.25, ge=0.0, le=1.0)

    #: Temporal window frequency used by ``pandas.Grouper`` (monthly default).
    temporal_window: str = "ME"

    #: Slope of the drift-vs-time regression that signals an increasing trend.
    temporal_slope_threshold: float = Field(default=0.01)

    #: Max p-value allowed for the temporal trend slope to be significant.
    temporal_trend_alpha: float = Field(default=0.10)

    #: Prediction entropy shift (in bits) considered significant.
    entropy_threshold: float = Field(default=0.20)

    #: Latitude/longitude grid size (degrees) used for spatial occupancy.
    spatial_cell_degrees: float = Field(default=1.0, gt=0.0)

    def severity_from_psi(self, value: float) -> str:
        """Map a PSI value to a low/moderate/high severity label."""
        if value < self.psi_thresholds[0]:
            return "low"
        if value < self.psi_thresholds[1]:
            return "moderate"
        return "high"

    def severity_from_js(self, value: float) -> str:
        """Map a JS divergence to a low/moderate/high severity label."""
        if value < self.js_thresholds[0]:
            return "low"
        if value < self.js_thresholds[1]:
            return "moderate"
        return "high"
