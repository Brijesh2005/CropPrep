"""Unit tests for the temporal index (season calendar + date helpers)."""

from __future__ import annotations

from datetime import date

import pytest

from services.spatial_alignment.config import DEFAULT_SEASONS, SeasonDef
from services.spatial_alignment.temporal_index import SeasonCalendar, TemporalIndex


@pytest.fixture
def calendar():
    return SeasonCalendar([SeasonDef(**s) for s in DEFAULT_SEASONS])


def test_kharif_window(calendar):
    season = calendar.season_window("Kharif", 2020)
    assert season.start == date(2020, 6, 1)
    assert season.end == date(2020, 10, 31)
    assert season.year == 2020
    assert not season.crosses_year_boundary


def test_rabi_window_crosses_year(calendar):
    season = calendar.season_window("Rabi", 2020)
    assert season.start == date(2020, 11, 1)
    assert season.end == date(2021, 3, 31)
    assert season.crosses_year_boundary


def test_summer_window(calendar):
    season = calendar.season_window("Summer", 2021)
    assert season.start == date(2021, 4, 1)
    assert season.end == date(2021, 5, 31)


def test_season_for_date_kharif(calendar):
    result = calendar.season_for_date(date(2020, 8, 15))
    assert result is not None
    season, year = result
    assert season.name == "Kharif"
    assert year == 2020


def test_season_for_date_rabi_january(calendar):
    # A January date belongs to the Rabi that started the previous November.
    result = calendar.season_for_date(date(2021, 1, 15))
    assert result is not None
    season, year = result
    assert season.name == "Rabi"
    assert season.year == 2020  # planting year


def test_season_for_date_rabi_december(calendar):
    result = calendar.season_for_date(date(2020, 12, 15))
    assert result is not None
    season, _ = result
    assert season.name == "Rabi"
    assert season.year == 2020


def test_season_for_date_gap_month(calendar):
    # June 5 is outside Rabi, inside Kharif.
    result = calendar.season_for_date(date(2020, 6, 5))
    assert result is not None
    assert result[0].name == "Kharif"


def test_unknown_season_raises(calendar):
    with pytest.raises(KeyError):
        calendar.season_window("Monsoon", 2020)


def test_sort_unique():
    dates = [date(2020, 7, 1), date(2020, 6, 1), date(2020, 7, 1), date(2020, 9, 1)]
    assert TemporalIndex.sort_unique(dates) == [
        date(2020, 6, 1), date(2020, 7, 1), date(2020, 9, 1),
    ]


def test_nearest_date():
    dates = [date(2020, 7, 1), date(2020, 8, 15), date(2020, 9, 1)]
    best, idx = TemporalIndex.nearest(date(2020, 8, 20), dates)
    assert best == date(2020, 8, 15)
    assert idx == 1


def test_gaps():
    dates = [date(2020, 6, 1), date(2020, 6, 15), date(2020, 9, 1)]
    assert TemporalIndex.gaps(dates) == [14.0, 78.0]
    assert TemporalIndex.max_gap_days(dates) == 78.0


def test_in_range():
    dates = [date(2020, 6, 1), date(2020, 7, 15), date(2020, 9, 1)]
    result = TemporalIndex.in_range(date(2020, 7, 1), date(2020, 8, 1), dates)
    assert result == [date(2020, 7, 15)]


def test_within_tolerance():
    assert TemporalIndex.within_tolerance(date(2020, 7, 16), date(2020, 7, 1), 15) is True
    assert TemporalIndex.within_tolerance(date(2020, 7, 20), date(2020, 7, 1), 15) is False


def test_dedupe_by_date():
    records = [
        {"observation_date": date(2020, 7, 1), "id": 1},
        {"observation_date": date(2020, 7, 1), "id": 2},
        {"observation_date": date(2020, 8, 1), "id": 3},
    ]
    result = TemporalIndex.dedupe_by_date(records)
    assert [r["id"] for r in result] == [1, 3]
