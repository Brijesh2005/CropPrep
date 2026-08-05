"""Quality control for STAM observations.

:func:`assess_quality` turns the pieces of an :class:`AgriculturalObservation`
into a :class:`QualityReport` with an overall score (0-100) and per-issue
detail. Checks cover:

* invalid coordinates,
* low-confidence location (distance above threshold),
* missing / unmatched tabular records,
* missing images and missing NDVI/EVI sides,
* temporal gaps and duplicate observation dates,
* CRS / resolution mismatches surfaced during pairing.

Scores are computed by subtracting per-severity penalties from 100.
"""

from __future__ import annotations

from typing import Sequence

from .config import QualityConfig, StamConfig
from .logger import get_logger
from .observation import (
    LocationInfo,
    QualityIssue,
    QualityReport,
    SequenceInfo,
    TabularFeatures,
    TemporalInfo,
)

logger = get_logger("validators")

#: Penalty per issue severity (applied once per issue).
_PENALTIES = {"critical": 40, "error": 25, "warning": 10, "info": 2}


def assess_quality(
    *,
    config: QualityConfig,
    location: LocationInfo,
    temporal: TemporalInfo,
    tabular: TabularFeatures | None,
    sequence: SequenceInfo,
    distance_threshold_km: float = 5.0,
    additional_issues: Sequence[QualityIssue] = (),
) -> QualityReport:
    """Build a :class:`QualityReport` for an assembled observation."""
    issues: list[QualityIssue] = []

    # -- Coordinates ---------------------------------------------------------- #
    if not (-180.0 <= location.lon <= 180.0 and -90.0 <= location.lat <= 90.0):
        issues.append(
            QualityIssue(
                code="ST-Q-COORD-001",
                severity="critical",
                message=f"Invalid coordinates: lon={location.lon}, lat={location.lat}",
            )
        )

    # -- Location confidence -------------------------------------------------- #
    if location.distance_km is not None and location.distance_km > distance_threshold_km:
        issues.append(
            QualityIssue(
                code="ST-Q-LOC-001",
                severity="warning",
                message=(
                    f"Nearest dataset location is {location.distance_km:.2f} km away "
                    f"(threshold {distance_threshold_km:.1f} km)"
                ),
                detail={"distance_km": location.distance_km},
            )
        )

    # -- Tabular -------------------------------------------------------------- #
    if tabular is None or tabular.matched_level == "none":
        issues.append(
            QualityIssue(
                code="ST-Q-TAB-001",
                severity="error",
                message="No exact tabular record matched for location/season/year",
                detail={"matched_level": tabular.matched_level if tabular else None},
            )
        )

    # -- Images --------------------------------------------------------------- #
    if not sequence.pairs:
        issues.append(
            QualityIssue(
                code="ST-Q-IMG-001",
                severity="error",
                message="No image observations found for the season/year",
            )
        )
    else:
        missing_sides = sum(
            1 for p in sequence.pairs if p.ndvi is None or p.evi is None
        )
        if missing_sides:
            issues.append(
                QualityIssue(
                    code="ST-Q-IMG-002",
                    severity="warning",
                    message=f"{missing_sides} observation date(s) missing NDVI or EVI",
                    detail={"count": missing_sides},
                )
            )

    # -- Temporal gaps -------------------------------------------------------- #
    max_gap = config.max_temporal_gap_days
    for gap in sequence.gap_days:
        if gap > max_gap:
            issues.append(
                QualityIssue(
                    code="ST-Q-TEMP-001",
                    severity="warning",
                    message=f"Temporal gap of {gap:.0f} days exceeds {max_gap} days",
                    detail={"gap_days": gap},
                )
            )

    if len(sequence.sorted_dates) < config.min_observations:
        issues.append(
            QualityIssue(
                code="ST-Q-TEMP-002",
                severity="error",
                message=(
                    f"Only {len(sequence.sorted_dates)} observation(s); "
                    f"minimum {config.min_observations} required"
                ),
            )
        )

    # -- Pairing mismatches (CRS / resolution / bbox) -------------------------- #
    for pair in sequence.pairs:
        if pair.quality.get("invalid"):
            issues.append(
                QualityIssue(
                    code="ST-Q-PAIR-001",
                    severity="error",
                    message=f"Invalid NDVI/EVI pairing on {pair.date}",
                    detail=pair.quality,
                )
            )

    issues.extend(additional_issues)

    # -- Score ----------------------------------------------------------------- #
    score = 100.0 - sum(_PENALTIES.get(i.severity, 0) for i in issues)
    score = max(0.0, min(100.0, score))
    failing = any(i.severity in {"critical", "error"} for i in issues)
    passed = (not failing) and score >= config.fail_below

    report = QualityReport(passed=passed, overall_score=round(score, 2), issues=issues)
    logger.info(
        "Quality assessed",
        extra={"passed": report.passed, "score": report.overall_score},
    )
    return report
