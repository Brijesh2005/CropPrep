"""Temporal imagery-window resolution and real-frame selection.

STAM historically matched NDVI/EVI records that fall inside the cropping-season
calendar window (Kharif Jun–Oct, Rabi Nov–Mar, Summer Apr–May) — see
:meth:`~training.stam.matcher.SpatialTemporalMatcher.match_images`. For
seasonal-composite remote-sensing datasets such as the Kaggle NDVI/EVI mount
(~5 composite dates per year clustered in late-Apr/May and late-Oct), that
season window resolves *exactly one* composite for Kharif surveys, starving the
deep temporal feature to a single real frame with ``T-1`` zero-filled slots.

This module makes the acquisition window explicit and configurable::

    * ``season``       — legacy behaviour (the season calendar window).
    * ``window_days``  — ``[survey_date - days, survey_date + days]``.
    * ``crop_year``    — ``[start_month .. start_month+span_months)`` of the
      survey reference's calendar year (season-agnostic crop-year context).

Once candidate records are collected, :func:`select_temporal_frames` trims the
ordered sequence to at most ``max_observations`` *real* frames using one of the
documented strategies. No date is fabricated, duplicated or zero-filled here —
a trim simply keeps a subset of the records that already exist on disk.

The module is deliberately framework-free (pure functions over
:class:`~training.stam.observation.ImagePairRef` lists) so the selection logic
is unit-testable against fabricated catalogs without rasters.
"""

from __future__ import annotations

import calendar as _calendar
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Callable, Sequence

from .config import ImageryWindowConfig
from .observation import ImagePairRef, SequenceInfo

__all__ = [
    "ImageryWindow",
    "resolve_window",
    "in_window",
    "select_temporal_frames",
    "pair_quality_score",
    "sequence_from_pairs",
    "pair_metadata",
    "window_description",
]

#: Delta between two dates in whole days (deterministic, bound).
def _days_between(left: date, right: date) -> int:
    return (right - left).days


@dataclass(frozen=True, slots=True)
class ImageryWindow:
    """A resolved acquisition window ``[start, end]`` (inclusive)."""

    start: date
    end: date
    mode: str
    description: str

    def contains(self, d: date) -> bool:
        return self.start <= d <= self.end


def _add_months(base: date, months: int) -> date:
    """Advance ``base`` by ``months`` calendar months (clamped day)."""
    month_index = base.year * 12 + (base.month - 1) + months
    year, month0 = divmod(month_index, 12)
    month = month0 + 1
    day = min(base.day, _calendar.monthrange(year, month)[1])
    return date(year, month, day)


def window_description(config: ImageryWindowConfig, reference_date: date | None = None) -> str:
    """A human-readable description of a configured imagery window."""
    if config.mode == "season":
        return "season-calendar window"
    if config.mode == "window_days":
        anchor = reference_date.isoformat() if reference_date else "survey_date"
        return f"+/-{config.window_days} days around {anchor}"
    return (
        f"crop-year window from month {config.start_month} spanning "
        f"{config.span_months} months of the survey year"
    )


def resolve_window(
    config: ImageryWindowConfig,
    *,
    reference_date: date | None,
    year: int | None = None,
    season=None,
) -> ImageryWindow:
    """Resolve the inclusive acquisition window for an observation.

    Args:
        config: The ``imagery`` configuration block.
        reference_date: Survey reference date (used by ``window_days`` and
            ``crop_year`` anchors). May be None when the season path is used.
        year: The observation's calendar year (fallback anchor).
        season: Resolved ``Season`` object (required for ``season`` mode).

    Raises:
        ValueError: When a mode needs a reference that is not available.
    """
    if config.mode == "season":
        if season is None:
            raise ValueError(
                "Imagery mode 'season' requires a resolved season; pass year+season"
            )
        return ImageryWindow(
            start=season.start,
            end=season.end,
            mode="season",
            description=window_description(config),
        )

    if config.mode == "window_days":
        if reference_date is None:
            raise ValueError(
                "Imagery mode 'window_days' requires a survey reference_date"
            )
        delta = timedelta(days=config.window_days)
        start = reference_date - delta
        end = reference_date + delta
        return ImageryWindow(
            start=start,
            end=end,
            mode="window_days",
            description=window_description(config, reference_date),
        )

    # crop_year — anchored on the reference year, else the observation year.
    anchor = (reference_date or date(year or 2000, 1, 1)).year
    start = date(anchor, config.start_month, 1)
    end = _add_months(start, config.span_months) - timedelta(days=1)
    return ImageryWindow(
        start=start,
        end=end,
        mode="crop_year",
        description=window_description(config, reference_date),
    )


def in_window(window: ImageryWindow, d: date | None, *, year: int | None = None) -> bool:
    """True when ``d`` falls inside ``window`` (undated records keep by year)."""
    if d is None:
        # Undated legacy records: fall back to the requested year membership.
        return year is not None and window.start.year <= year <= window.end.year
    return window.contains(d)


def pair_quality_score(pair: ImagePairRef) -> float:
    """A continuous per-frame quality score in ``[0, 2]``.

    Not a prediction or a learned value — a deterministic aggregation of the
    sequence builder's already-recorded pairing facts:
    ``paired`` (clean both-index frame), missing-stream penalties and the
    invalid (resolution/CRS/bounds mismatch) flag.
    """
    quality = getattr(pair, "quality", None) or {}
    score = 0.0
    if pair.ndvi is not None:
        score += 1.0
    if pair.evi is not None:
        score += 1.0
    if quality.get("paired"):
        score += 0.0  # fully reported by the stream presence above
    for missing in quality.get("missing", []):
        score -= 0.0  # per-stream score already reflects absence
    if quality.get("invalid"):
        score = max(0.0, score - 1.0)
    return round(score, 4)


def _rank_indexes(
    pairs: Sequence[ImagePairRef],
    score_fn: Callable[[ImagePairRef, int], float],
) -> list[int]:
    """Indexes sorted by ``score_fn`` descending, ties by position."""
    ranked = sorted(
        range(len(pairs)), key=lambda i: (-score_fn(pairs[i], i), i)
    )
    return ranked


def _select_indexes(
    n_dates: int,
    reference_date: date | None,
    config: ImageryWindowConfig,
    pairs: Sequence[ImagePairRef],
    dates: Sequence[date],
) -> list[int]:
    """Choose up to ``max_observations`` date indexes from the window set."""
    keep = min(config.max_observations, n_dates)
    if keep <= 0:
        return []
    if keep >= n_dates:
        return list(range(n_dates))

    strategy = config.strategy

    if strategy in ("closest_to_survey",):
        if reference_date is None:
            ranked = _rank_indexes(pairs, lambda p, i: pair_quality_score(p))
        else:
            ranked = _rank_indexes(
                pairs,
                lambda p, i: -abs(_days_between(p.date, reference_date)),
            )
        chosen = ranked[:keep]
    elif strategy == "quality_ranked":
        chosen = _rank_indexes(pairs, lambda p, i: pair_quality_score(p))[:keep]
    elif strategy == "temporal_quality_combined":
        span = max(config.window_days, 1)

        def combined(pair: ImagePairRef, i: int) -> float:
            quality = pair_quality_score(pair)
            proximity = (
                0.0
                if reference_date is None
                else abs(_days_between(pair.date, reference_date)) / span
            )
            return quality - 0.02 * proximity

        chosen = _rank_indexes(pairs, combined)[:keep]
    elif strategy == "phenology_coverage":
        # Bucket the available dates into coarse phenology phases (2-month bins)
        # and pick representatives spread across the crop-year cycle.
        bins: dict[tuple[int, int], list[date]] = {}
        for idx, d in enumerate(dates):
            bins.setdefault((d.year, (d.month - 1) // 2), []).append(idx)
        ordered_bins = sorted(bins)
        if keep <= len(ordered_bins):
            step = len(ordered_bins) / keep
            chosen = [
                int(bins[ordered_bins[min(len(ordered_bins) - 1, int(round(i * step)))]][0])
                for i in range(keep)
            ]
        else:
            chosen = []
            for _, idxs in bins.items():
                chosen.append(idxs[0])
            # Fill the remainder evenly across remaining slot positions.
            fill = [(dates[i], i) for i in range(n_dates) if i not in set(chosen)]
            fill.sort()
            for i in range(keep - len(chosen)):
                chosen.append(fill[min(len(fill) - 1, i)][1])
    else:  # evenly_spaced
        step = (n_dates - 1) / max(keep - 1, 1)
        chosen = {round(i * step) for i in range(keep)}

    unique = sorted(set(chosen), key=lambda i: dates[i]) if chosen else []
    return unique[:keep]


def select_temporal_frames(
    pairs: Sequence[ImagePairRef],
    reference_date: date | None,
    config: ImageryWindowConfig,
) -> list[ImagePairRef]:
    """Trim an ordered sequence to ``<= max_observations`` real frames.

    The input ``pairs`` are expected to be ordered by date (the sequence
    builder's output). Frames are *selected*, never synthesised: indexes are
    chosen from the real dates present, then the result is re-sorted by date so
    the temporal order is preserved.

    Args:
        pairs: Full window sequence (real records only).
        reference_date: Survey reference date (strategy anchor).
        config: The ``imagery`` configuration (``max_observations``/strategy).

    Returns:
        Up to ``max_observations`` pairs ordered by date.
    """
    dates = [p.date for p in pairs]
    n = len(dates)
    if n == 0:
        return []
    selected = _select_indexes(n, reference_date, config, list(pairs), dates)
    kept = [pairs[i] for i in selected]
    kept.sort(key=lambda p: p.date)
    # Attach frame metadata for provenance and diagnostics.
    for pair in kept:
        pair.quality.update(pair_metadata(pair, reference_date))
    return kept


def sequence_from_pairs(
    original: SequenceInfo,
    kept: Sequence[ImagePairRef],
) -> SequenceInfo:
    """Rebuild a :class:`SequenceInfo` from a subset of its pairs.

    Recomputes ``sorted_dates`` and ``gap_days``; transfers resolution/CRS and
    the original record paths so the object stays a valid standalone sequence.
    """
    kept_list = sorted(kept, key=lambda p: p.date)
    sorted_dates = [p.date for p in kept_list]
    gaps: list[float] = []
    for left, right in zip(sorted_dates, sorted_dates[1:]):
        gaps.append(float(_days_between(left, right)))
    return SequenceInfo(
        pairs=list(kept_list),
        sorted_dates=sorted_dates,
        resolution=original.resolution,
        crs=original.crs,
        ndvi_paths=list(original.ndvi_paths),
        evi_paths=list(original.evi_paths),
        gap_days=gaps,
    )


def pair_metadata(pair: ImagePairRef, reference_date: date | None) -> dict[str, Any]:
    """Per-frame metadata stamped into ``pair.quality`` for diagnostics."""
    meta: dict[str, Any] = {
        "image_date": pair.date.isoformat(),
        "ndvi_available": pair.ndvi is not None,
        "evi_available": pair.evi is not None,
    }
    if reference_date is not None:
        meta["days_from_survey"] = _days_between(pair.date, reference_date)
    return meta