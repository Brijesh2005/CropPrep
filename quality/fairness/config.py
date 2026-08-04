"""Fairness thresholds and configuration."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FairnessConfig(BaseModel):
    """Acceptable thresholds for each fairness metric.

    Thresholds follow the commonly-used 80% rule for disparate impact and
    0.1 (10 percentage points) for parity differences.
    """

    model_config = {"extra": "forbid"}

    #: Minimum acceptable disparate impact ratio (80% rule).
    disparate_impact_min: float = Field(default=0.80, gt=0.0, le=1.0)

    #: Max acceptable statistical-parity difference (absolute pp).
    statistical_parity_max: float = Field(default=0.10, gt=0.0)

    #: Max acceptable equalized-odds difference (max |ΔTPR|, |ΔFPR|).
    equalized_odds_max: float = Field(default=0.10, gt=0.0)

    #: Max acceptable equal-opportunity difference (|ΔTPR|).
    equal_opportunity_max: float = Field(default=0.10, gt=0.0)

    #: Max acceptable accuracy difference across groups.
    accuracy_parity_max: float = Field(default=0.10, gt=0.0)

    #: Max acceptable calibration (ECE) difference across groups.
    calibration_parity_max: float = Field(default=0.05, gt=0.0)

    #: Bins used to compute Expected Calibration Error.
    calibration_bins: int = Field(default=10, ge=2)

    #: Minimum group size before it is reported as insufficient.
    min_group_size: int = Field(default=30, ge=1)

    #: Severity escalation when a threshold is exceeded by this multiple.
    severity_multiplier: float = Field(default=2.0, ge=1.0)
