"""Ordered NDVI/EVI observation-sequence builder.

:class:`ImagePairBuilder` pairs NDVI and EVI records that share an observation
date after validating same resolution, same CRS and same bounding box.
:class:`ObservationSequenceBuilder` assembles the full ordered time series for
a (location, year, season) — handling duplicate dates, out-of-order dates,
missing observations and per-date quality flags.

The sequence holds *references* (paths + metadata) only. Pixel arrays are read
lazily via :class:`~services.spatial_alignment.patch_generator.SpatialPatchGenerator`
so no raster is ever loaded implicitly.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Iterable, Sequence

from .exceptions import PairingError
from .logger import get_logger
from .observation import ImagePairRef, ImageRecordRef, QualityIssue, SequenceInfo
from .temporal_index import TemporalIndex

logger = get_logger("sequence_builder")


@dataclass(slots=True)
class SequenceBuildResult:
    """Outcome of building an ordered image sequence."""

    sequence: SequenceInfo
    issues: list[QualityIssue] = field(default_factory=list)
    ndvi_count: int = 0
    evi_count: int = 0
    paired_count: int = 0
    duplicate_dates: int = 0

    @property
    def observation_count(self) -> int:
        return len(self.sequence.pairs)


class ImagePairBuilder:
    """Pair NDVI and EVI records by observation date with validation."""

    def __init__(self, *, require_pairs: bool = True) -> None:
        self.require_pairs = require_pairs

    def build(
        self,
        ndvi_records: Sequence[ImageRecordRef],
        evi_records: Sequence[ImageRecordRef],
    ) -> tuple[list[ImagePairRef], list[QualityIssue], int]:
        """Pair records by date; returns ``(pairs, issues, duplicate_count)``.

        Pairs are validated for matching resolution, CRS and bounding box. A
        date with only one index still yields a (partial) pair carrying a
        ``missing`` quality flag so the caller can decide how to weight it.
        """
        ndvi_by_date, ndvi_dups = _unique_by_date(ndvi_records, "NDVI")
        evi_by_date, evi_dups = _unique_by_date(evi_records, "EVI")
        issues: list[QualityIssue] = ndvi_dups + evi_dups
        duplicate_count = len(ndvi_dups) + len(evi_dups)

        dates = sorted(set(ndvi_by_date) | set(evi_by_date))
        pairs: list[ImagePairRef] = []
        for day in dates:
            ndvi = ndvi_by_date.get(day)
            evi = evi_by_date.get(day)
            missing: list[str] = []
            if ndvi is None:
                missing.append("NDVI")
            if evi is None:
                missing.append("EVI")

            pair_issues: list[QualityIssue] = []
            if ndvi is not None and evi is not None:
                pair_issues.extend(_validate_pair(ndvi, evi))

            quality: dict = {
                "paired": ndvi is not None and evi is not None and not pair_issues,
                "missing": missing,
                "invalid": bool(pair_issues),
            }
            if pair_issues:
                issues.extend(pair_issues)

            ref = ndvi or evi
            pairs.append(
                ImagePairRef(
                    date=day,
                    ndvi=ndvi,
                    evi=evi,
                    resolution=ref.resolution if ref else "UNKNOWN",
                    crs=ref.crs if ref else None,
                    quality=quality,
                )
            )

        if self.require_pairs and any(p.ndvi is None or p.evi is None for p in pairs):
            raise PairingError(
                "Required NDVI/EVI pairing failed for one or more dates",
                detail=[p.date.isoformat() for p in pairs if p.ndvi is None or p.evi is None],
            )
        return pairs, issues, duplicate_count


class ObservationSequenceBuilder:
    """Build the ordered NDVI/EVI time series for an observation."""

    def __init__(
        self,
        *,
        require_pairs: bool = True,
        max_gap_days: int = 60,
    ) -> None:
        self.pair_builder = ImagePairBuilder(require_pairs=require_pairs)
        self.max_gap_days = max_gap_days

    def build(
        self,
        ndvi_records: Iterable[ImageRecordRef],
        evi_records: Iterable[ImageRecordRef],
        *,
        resolution: str | None = None,
    ) -> SequenceBuildResult:
        """Assemble an ordered sequence from NDVI + EVI record lists.

        Steps: dedupe per index → pair by date → sort by date → detect gaps.
        """
        ndvi_list = list(ndvi_records)
        evi_list = list(evi_records)

        pairs, issues, duplicates = self.pair_builder.build(ndvi_list, evi_list)

        # Sorted by date (out-of-order inputs are normalised here).
        pairs.sort(key=lambda p: p.date)
        sorted_dates = [p.date for p in pairs]
        gap_days = TemporalIndex.gaps(sorted_dates)

        resolution = resolution or _resolve_resolution(ndvi_list, evi_list)
        crs = _resolve_crs(ndvi_list, evi_list)

        # Gap detection beyond the tolerance.
        for left, right in zip(sorted_dates, sorted_dates[1:]):
            gap = (right - left).days
            if gap > self.max_gap_days:
                issues.append(
                    QualityIssue(
                        code="ST-TEMP-002",
                        severity="warning",
                        message=f"Temporal gap of {gap} days between {left} and {right}",
                        detail={"left": left.isoformat(), "right": right.isoformat()},
                    )
                )

        sequence = SequenceInfo(
            pairs=pairs,
            sorted_dates=sorted_dates,
            resolution=resolution,
            crs=crs,
            ndvi_paths=[p.path for p in ndvi_list],
            evi_paths=[p.path for p in evi_list],
            gap_days=gap_days,
        )
        result = SequenceBuildResult(
            sequence=sequence,
            issues=issues,
            ndvi_count=len(ndvi_list),
            evi_count=len(evi_list),
            paired_count=sum(1 for p in pairs if p.ndvi is not None and p.evi is not None),
            duplicate_dates=duplicates,
        )
        logger.info(
            "Sequence built",
            extra={
                "ndvi": len(ndvi_list),
                "evi": len(evi_list),
                "pairs": len(pairs),
                "issues": len(issues),
            },
        )
        return result


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _unique_by_date(
    records: Sequence[ImageRecordRef], index: str
) -> tuple[dict[date, ImageRecordRef], list[QualityIssue]]:
    """First-record-wins deduplication by observation date."""
    by_date: dict[date, ImageRecordRef] = {}
    issues: list[QualityIssue] = []
    for record in records:
        if record.observation_date is None:
            issues.append(
                QualityIssue(
                    code="ST-IMAGE-002",
                    severity="warning",
                    message=f"{index} record has no observation date",
                    detail=record.relative_path,
                )
            )
            continue
        if record.observation_date in by_date:
            issues.append(
                QualityIssue(
                    code="ST-IMAGE-003",
                    severity="warning",
                    message=f"Duplicate {index} observation on {record.observation_date}",
                    detail=record.relative_path,
                )
            )
            continue
        by_date[record.observation_date] = record
    return by_date, issues


def _validate_pair(ndvi: ImageRecordRef, evi: ImageRecordRef) -> list[QualityIssue]:
    """Validate that two records can be paired (resolution/CRS/bounds)."""
    issues: list[QualityIssue] = []
    if ndvi.resolution != evi.resolution:
        issues.append(
            QualityIssue(
                code="ST-PAIR-002",
                severity="error",
                message=f"Resolution mismatch on {ndvi.observation_date}: "
                        f"{ndvi.resolution} vs {evi.resolution}",
                detail={"ndvi": ndvi.path, "evi": evi.path},
            )
        )
    if (ndvi.crs or "").lower() != (evi.crs or "").lower():
        issues.append(
            QualityIssue(
                code="ST-PAIR-003",
                severity="error",
                message=f"CRS mismatch on {ndvi.observation_date}: "
                        f"{ndvi.crs} vs {evi.crs}",
                detail={"ndvi": ndvi.path, "evi": evi.path},
            )
        )
    if ndvi.bounds != evi.bounds:
        issues.append(
            QualityIssue(
                code="ST-PAIR-004",
                severity="error",
                message=f"Bounding-box mismatch on {ndvi.observation_date}",
                detail={"ndvi": ndvi.bounds, "evi": evi.bounds},
            )
        )
    return issues


def _resolve_resolution(
    ndvi: Sequence[ImageRecordRef], evi: Sequence[ImageRecordRef]
) -> str:
    counter: dict[str, int] = defaultdict(int)
    for record in [*ndvi, *evi]:
        counter[record.resolution] += 1
    return max(counter, key=counter.get) if counter else "UNKNOWN"


def _resolve_crs(
    ndvi: Sequence[ImageRecordRef], evi: Sequence[ImageRecordRef]
) -> str | None:
    for record in [*ndvi, *evi]:
        if record.crs:
            return record.crs
    return None
