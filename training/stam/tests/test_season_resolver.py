"""Tests for the SeasonResolver (date -> season mapping).

The resolver backs the farmer workflow: a farmer supplies only a location and
the system infers the season from the calendar date using the YAML-calendar.
"""

from __future__ import annotations

from datetime import date

from training.stam import SeasonResolver, StamConfig


def test_resolve_kharif():
    resolver = SeasonResolver.from_config(today=date(2026, 8, 15))
    season, year = resolver.resolve(date(2026, 8, 15))
    assert season.name == "Kharif"
    assert year == 2026
    assert resolver.season_name(date(2026, 8, 15)) == "Kharif"


def test_resolve_rabi_january_uses_previous_planting_year():
    # Jan 2026 falls in the Rabi season that started Nov 2025.
    resolver = SeasonResolver.from_config(today=date(2026, 1, 15))
    season, year = resolver.resolve(date(2026, 1, 15))
    assert season.name == "Rabi"
    assert year == 2025
    assert season.crosses_year_boundary is True


def test_resolve_rabi_november_starts_new_planting_year():
    resolver = SeasonResolver.from_config(today=date(2025, 11, 10))
    season, year = resolver.resolve(date(2025, 11, 10))
    assert season.name == "Rabi"
    assert year == 2025


def test_resolve_summer():
    resolver = SeasonResolver.from_config(today=date(2026, 4, 20))
    assert resolver.season_name(date(2026, 4, 20)) == "Summer"


def test_every_month_maps_to_a_season():
    resolver = SeasonResolver.from_config(today=date(2026, 1, 1))
    for month in range(1, 13):
        assert resolver.resolve(date(2026, month, 15)) is not None


def test_names_and_stable_version():
    resolver = SeasonResolver.from_config()
    assert resolver.names() == ["Kharif", "Rabi", "Summer"]
    assert resolver.version.startswith("1.")
    assert len(resolver.version) == len("1.") + 8


def test_calendar_matches_stam_config_defaults():
    resolver = SeasonResolver.from_config(StamConfig())
    assert [d.name for d in resolver.definitions] == ["Kharif", "Rabi", "Summer"]


def test_ignores_missing_file_and_falls_back_to_config(tmp_path):
    resolver = SeasonResolver.from_config(
        StamConfig(), seasons_file=tmp_path / "nope.yaml", today=date(2026, 6, 15)
    )
    assert resolver.resolve(date(2026, 6, 15))[0].name == "Kharif"
