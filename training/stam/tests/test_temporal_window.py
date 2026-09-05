"""Tests for the temporal imagery-window module (R5.3).

Pure-function tests over fabricated ImagePairRef/SequenceInfo objects — no
rasters, no Dataset Manager. Covers window resolution (season / window_days /
crop_year), frame-selection strategies, trimming to max_observations, metadata
stamping and sequence reconstruction.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from training.stam.config import ImageryWindowConfig
from training.stam.observation import (
    ImagePairRef,
    ImageRecordRef,
    SequenceInfo,
)
from training.stam.temporal_index import Season
from training.stam.temporal_window import (
    ImageryWindow,
    in_window,
    pair_metadata,
    pair_quality_score,
    resolve_window,
    select_temporal_frames,
    sequence_from_pairs,
    window_description,
)

SURVEY = date(2020, 9, 15)


def _record(index_type: str, d: date, resolution: str = "R10m") -> ImageRecordRef:
    return ImageRecordRef(
        path=f"/data/{d.isoformat()}_{index_type}.tif",
        relative_path=f"{d.isoformat()}_{index_type}.tif",
        index_type=index_type,
        resolution=resolution,
        observation_date=d,
        year=d.year,
        crs="EPSG:32643",
        pixel_size=(10.0, 10.0),
        bounds=(0.0, 0.0, 100.0, 100.0),
    )


def _pair(d: date, *, paired: bool = True) -> ImagePairRef:
    ndvi = _record("NDVI", d)
    evi = _record("EVI", d) if paired else None
    return ImagePairRef(
        date=d,
        ndvi=ndvi,
        evi=evi,
        resolution="R10m",
        crs="EPSG:32643",
        quality={
            "paired": paired,
            "missing": [] if paired else ["EVI"],
            "invalid": False,
        },
    )


def _sequence(dates: list[date]) -> SequenceInfo:
    pairs = [_pair(d) for d in dates]
    return SequenceInfo(
        pairs=pairs,
        sorted_dates=sorted(dates),
        resolution="R10m",
        crs="EPSG:32643",
        ndvi_paths=[p.ndvi.path for p in pairs],
        evi_paths=[p.evi.path for p in pairs],
        gap_days=[0.0],
    )


def _config(**overrides) -> ImageryWindowConfig:
    return ImageryWindowConfig(**overrides)


class TestResolveWindow:
    def test_season_mode_uses_season_bounds(self) -> None:
        cfg = _config(mode="season")
        season = Season(year=2020, name="Kharif", start=date(2020, 6, 1), end=date(2020, 10, 31))
        win = resolve_window(cfg, reference_date=SURVEY, year=2020, season=season)
        assert win == ImageryWindow(
            start=date(2020, 6, 1), end=date(2020, 10, 31), mode="season",
            description="season-calendar window",
        )
        assert win.contains(date(2020, 10, 28))
        assert not win.contains(date(2020, 5, 1))

    def test_season_mode_requires_season(self) -> None:
        with pytest.raises(ValueError):
            resolve_window(_config(mode="season"), reference_date=None, year=2020, season=None)

    def test_window_days_symmetric_around_reference(self) -> None:
        cfg = _config(mode="window_days", window_days=180)
        win = resolve_window(cfg, reference_date=SURVEY, year=2020)
        assert win.start == SURVEY - timedelta(days=180)
        assert win.end == SURVEY + timedelta(days=180)
        assert "+/-180 days" in win.description

    def test_window_days_requires_reference(self) -> None:
        with pytest.raises(ValueError):
            resolve_window(_config(mode="window_days"), reference_date=None)

    def test_crop_year_anchors_on_reference_year(self) -> None:
        cfg = _config(mode="crop_year", start_month=5, span_months=12)
        win = resolve_window(cfg, reference_date=SURVEY, year=2020)
        assert win.start == date(2020, 5, 1)
        assert win.end == date(2021, 4, 30)
        assert win.mode == "crop_year"

    def test_crop_year_spans_year_boundary(self) -> None:
        cfg = _config(mode="crop_year", start_month=9, span_months=13)
        win = resolve_window(cfg, reference_date=date(2020, 12, 1), year=2020)
        assert win.start == date(2020, 9, 1)
        assert win.end == date(2021, 9, 30)


class TestInWindow:
    def test_undated_fallback_uses_year(self) -> None:
        window = ImageryWindow(date(2020, 1, 1), date(2020, 12, 31), "season", "x")
        assert in_window(window, None, year=2020)
        assert not in_window(window, None, year=2019)


class TestPairQuality:
    def test_paired_scores_2(self) -> None:
        assert pair_quality_score(_pair(date(2020, 10, 28))) == 2.0

    def test_missing_stream_scores_1(self) -> None:
        assert pair_quality_score(_pair(date(2020, 10, 28), paired=False)) == 1.0

    def test_invalid_penalised(self) -> None:
        pair = _pair(date(2020, 10, 28))
        pair.quality["invalid"] = True
        assert pair_quality_score(pair) == 1.0


class TestSelectTemporalFrames:
    DATES = [
        date(2019, 10, 9),
        date(2019, 10, 14),
        date(2019, 10, 29),
        date(2020, 4, 16),
        date(2020, 5, 1),
        date(2020, 5, 11),
        date(2020, 5, 26),
        date(2020, 10, 28),
    ]

    def _seq(self) -> SequenceInfo:
        return _sequence(self.DATES)

    def test_no_trim_when_within_max(self) -> None:
        kept = select_temporal_frames(self._seq().pairs, SURVEY, _config(max_observations=8))
        assert len(kept) == len(self.DATES)

    def test_trim_to_max_observations(self) -> None:
        kept = select_temporal_frames(
            self._seq().pairs, SURVEY, _config(max_observations=4, strategy="closest_to_survey")
        )
        assert len(kept) == 4

    def test_closest_to_survey_keeps_nearby_dates(self) -> None:
        # Survey 2020-09-15; closest frame is 2020-10-28 (43 days), next
        # 2020-05-26 then 2020-05-11. With max=3 the kept dates must be those.
        kept = select_temporal_frames(
            self._seq().pairs, SURVEY, _config(max_observations=3, strategy="closest_to_survey")
        )
        assert [p.date for p in kept] == [
            date(2020, 5, 11), date(2020, 5, 26), date(2020, 10, 28),
        ]

    def test_evenly_spaced_spreads_range(self) -> None:
        kept = select_temporal_frames(
            self._seq().pairs, SURVEY, _config(max_observations=4, strategy="evenly_spaced")
        )
        dates = [p.date for p in kept]
        assert dates[0] == self.DATES[0]  # earliest retained
        assert dates[-1] == self.DATES[-1]  # latest retained
        assert len(dates) == len(set(dates))

    def test_phenology_coverage_multiple_phases(self) -> None:
        kept = select_temporal_frames(
            self._seq().pairs, SURVEY, _config(max_observations=4, strategy="phenology_coverage")
        )
        phases = {(d.year, (d.month - 1) // 2) for d in [p.date for p in kept]}
        assert len(phases) >= 3  # spans at least 3 two-month phenology phases

    def test_quality_ranked_prefers_paired(self) -> None:
        seq = _sequence([date(2020, 10, 28)])
        seq.pairs = [
            _pair(date(2020, 5, 1), paired=False),
            _pair(date(2020, 10, 28), paired=True),
        ]
        kept = select_temporal_frames(
            seq.pairs, SURVEY, _config(max_observations=1, strategy="quality_ranked")
        )
        assert kept[0].date == date(2020, 10, 28)

    def test_empty_input_returns_empty(self) -> None:
        assert select_temporal_frames([], SURVEY, _config()) == []

    def test_metadata_stamped(self) -> None:
        kept = select_temporal_frames(
            self._seq().pairs, SURVEY, _config(max_observations=2, strategy="closest_to_survey")
        )
        meta = kept[0].quality
        assert meta["image_date"] == kept[0].date.isoformat()
        assert meta["ndvi_available"] is True
        assert "days_from_survey" in meta


class TestSequenceFromPairs:
    def test_rebuild_preserves_order_and_paths(self) -> None:
        original = _sequence(
            [date(2019, 10, 9), date(2020, 5, 1), date(2020, 10, 28)]
        )
        kept = [original.pairs[1], original.pairs[2]]  # out of order input
        rebuilt = sequence_from_pairs(original, kept)
        assert rebuilt.sorted_dates == [date(2020, 5, 1), date(2020, 10, 28)]
        assert rebuilt.resolution == "R10m"
        assert len(rebuilt.ndvi_paths) == 3
        assert len(rebuilt.evi_paths) == 3

    def test_gap_days_recomputed(self) -> None:
        original = _sequence([date(2020, 5, 1), date(2020, 5, 26), date(2020, 10, 28)])
        rebuilt = sequence_from_pairs(original, original.pairs)
        assert rebuilt.gap_days == [25.0, 155.0]


class TestWindowDescription:
    def test_season(self) -> None:
        assert window_description(_config(mode="season")) == "season-calendar window"

    def test_window_days_includes_anchor(self) -> None:
        desc = window_description(_config(mode="window_days", window_days=90), SURVEY)
        assert "+/-90" in desc and "2020-09-15" in desc

    def test_crop_year(self) -> None:
        desc = window_description(_config(mode="crop_year", start_month=5, span_months=12))
        assert "month 5" in desc and "12 months" in desc