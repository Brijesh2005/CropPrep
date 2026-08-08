"""End-to-end tests for the STAM facade over the synthetic dataset."""

from __future__ import annotations

import pytest

from training.stam.exceptions import (
    LocationNotFoundError,
    NotInitializedError,
)
from training.stam.name_aliases import district_to_csv, normalize_name, resolve_location
from training.stam.observation import AgriculturalObservation
from training.stam.stam import STAM


def test_alias_mapping():
    assert normalize_name("Mangalore") == "Dakshina Kannada"
    assert normalize_name("Bangalore") == "Bengaluru"
    assert normalize_name("Chikmangaluru") == "Chikkamagaluru"
    assert normalize_name("Davangere") == "Davanagere"
    assert normalize_name("Gulbarga") == "Kalaburgi"
    assert normalize_name("Kodagu") == "Kodagu"


def test_alias_mapping_icrisat_spellings():
    # ICRISAT keeps the pre-2014 district spellings; the alias layer maps
    # them onto the KGIS boundary spellings for the multi-table join.
    assert normalize_name("Belgaum") == "Belagavi"
    assert normalize_name("Bellary") == "Ballari"
    assert normalize_name("Bijapur") == "Vijayapura"
    assert normalize_name("Chickmagalur") == "Chikkamagaluru"
    assert normalize_name("Kolar") == "Kolara"
    assert normalize_name("Mysore") == "Mysuru"
    assert normalize_name("Shimoge") == "Shivamogga"
    assert normalize_name("Tumkur") == "Tumakuru"


def test_first_contains_slash_alternates():
    # ICRISAT stores renamed districts as "Gulbarga / Kalaburagi"; either
    # alternate must match the boundary's canonical name.
    import pandas as pd

    from training.stam.matcher import _first_contains

    frame = pd.DataFrame({"dist": ["Gulbarga / Kalaburagi", "Kodagu / Coorg"]})
    row = _first_contains(frame, "dist", "Kalaburgi")
    assert row is not None and row["dist"] == "Gulbarga / Kalaburagi"
    row = _first_contains(frame, "dist", "Kodagu")
    assert row is not None and row["dist"] == "Kodagu / Coorg"


def test_normalize_name_case_and_boundary_side():
    # The boundary side must land on the same canonical spelling.
    assert normalize_name("Dakshina Kannada") == "Dakshina Kannada"
    assert normalize_name("  mangalore  ") == "Dakshina Kannada"
    assert normalize_name("Bengaluru (Urban)") == "Bengaluru"
    assert normalize_name(None) == ""


def test_kasaragodu_handling():
    result = resolve_location("Kasaragodu")
    assert result is not None
    assert result.status in ["manual", "unmatched"]
    assert result.lon is not None and result.lat is not None


def test_not_initialized_raises(manager, stam_config):
    stam = STAM(manager, stam_config)
    with pytest.raises(NotInitializedError):
        stam.build_observation(74.802, 13.098, year=2020, season="Kharif")


def test_initialize_and_summary(stam):
    summary = stam.summary()
    assert summary["initialized"] is True
    assert summary["patch_size"] == 16
    assert summary["indexes"]["locations"] > 0
    assert summary["seasons"] == ["Kharif", "Rabi", "Summer"]


def test_build_observation_full(stam):
    obs = stam.build_observation(74.802, 13.098, year=2020, season="Kharif")
    assert isinstance(obs, AgriculturalObservation)
    assert obs.location.admin.village == "A"
    assert obs.temporal.year == 2020
    assert obs.temporal.season == "Kharif"
    assert obs.crop == "Rice"
    assert obs.yield_value == 5200.0
    assert obs.num_observations() == 3
    assert obs.has_paired_images is True
    assert obs.quality.passed is True
    assert obs.sequence.resolution == "R10m"
    assert len(obs.sequence.ndvi_paths) == 3
    assert len(obs.sequence.evi_paths) == 3


def test_build_observation_serializable(stam):
    obs = stam.build_observation(74.802, 13.098, year=2020, season="Kharif")
    payload = obs.model_dump(mode="json")
    assert payload["observation_id"]
    assert payload["temporal"]["year"] == 2020
    # Round-trips through the model.
    rebuilt = AgriculturalObservation.model_validate(payload)
    assert rebuilt.crop == "Rice"


def test_build_observation_train_dict(stam):
    obs = stam.build_observation(74.802, 13.098, year=2020, season="Kharif")
    train = obs.to_train_dict()
    assert train["crop"] == "Rice"
    assert train["yield_value"] == 5200.0
    assert len(train["ndvi_paths"]) == 3
    assert len(train["evi_paths"]) == 3
    assert train["quality_score"] == obs.quality.overall_score


def test_build_observation_year_default(stam):
    # Without year/season: year defaults to the latest tabular year (2021),
    # whose sequence has NDVI only -> strict pairing falls back and flags it.
    obs = stam.build_observation(74.802, 13.098)
    assert obs.temporal.year == 2021
    assert obs.quality.passed is False


def test_build_observation_farmer_path_resolves_season(stam):
    # Farmer path: location only — the season is inferred from the date and a
    # multi-year historical context (same location + same season) is attached.
    obs = stam.build_observation(74.802, 13.098)
    assert obs.temporal.season is not None
    assert obs.temporal.season in ["Kharif", "Rabi", "Summer"]
    assert obs.historical_context is not None
    assert obs.historical_context.season == obs.temporal.season
    assert obs.historical_context.resolved_year == obs.temporal.year
    assert obs.historical_context.years == [2019, 2020, 2021]
    assert obs.historical_context.total_records > 0
    assert obs.historical_context.source == "dataset_manager"
    assert obs.dataset_version == obs.historical_context.dataset_version
    assert obs.season_calendar_version == obs.historical_context.season_calendar_version
    assert obs.provenance["historical_context_years"] == [2019, 2020, 2021]


def test_build_observation_research_path_keeps_versions(stam):
    # Research path: explicit year/season still resolves the season and
    # records the calendar + dataset versions on the observation.
    obs = stam.build_observation(74.802, 13.098, year=2020, season="Kharif")
    assert obs.temporal.year == 2020
    assert obs.temporal.season == "Kharif"
    assert obs.season_calendar_version is not None
    assert obs.dataset_version is None or isinstance(obs.dataset_version, str)
    assert obs.historical_context.years == [2019, 2020, 2021]


def test_build_observation_missing_evi_flags(stam):
    obs = stam.build_observation(74.802, 13.098, year=2021, season="Kharif")
    assert obs.num_observations() == 1
    assert any(i.code == "ST-Q-PAIR-002" for i in obs.quality.issues)


def test_build_observation_location_not_found(stam):
    with pytest.raises(LocationNotFoundError):
        stam.build_observation(78.0, 20.0, year=2020, season="Kharif")


def test_find_nearest(stam):
    nearest = stam.find_nearest(74.802, 13.098)
    assert nearest["distance_km"] < 1.0
    assert nearest["id"]


def test_build_sequence(stam):
    result = stam.build_sequence(74.802, 13.098, year=2020, season="Kharif")
    assert result.observation_count == 3
    assert result.paired_count == 3
    assert result.sequence.sorted_dates[0].year == 2020


def test_get_patch(stam):
    obs = stam.build_observation(74.802, 13.098, year=2020, season="Kharif")
    ndvi_path = obs.sequence.pairs[0].ndvi.path
    patch = stam.get_patch(ndvi_path, 74.802, 13.098, size=16)
    assert patch.shape == (16, 16)
    assert patch.valid_ratio > 0.5
    assert patch.crs == "EPSG:4326"


def test_validate_reports(stam):
    obs = stam.build_observation(74.802, 13.098, year=2020, season="Kharif")
    report = stam.validate(obs)
    assert report.passed is True
    assert report.overall_score == obs.quality.overall_score


def test_observation_cache_hit(stam, manager):
    stam.build_observation(74.802, 13.098, year=2020, season="Kharif")
    key = stam.cache.observation_key(74.802, 13.098, 2020, "Kharif")
    cached = manager.cache_get(key)
    assert cached is not None
    assert cached["crop"] == "Rice"


def test_find_nearest_before_initialize_raises(manager, stam_config):
    stam = STAM(manager, stam_config)
    with pytest.raises(NotInitializedError):
        stam.find_nearest(74.802, 13.098)


# --------------------------------------------------------------------------- #
# Multi-source tabular fallback (data_season-like -> ICRISAT-like chain)
# --------------------------------------------------------------------------- #


def _multi_table_stam(manager, stam_config_multi_table):
    stam = STAM(manager, stam_config_multi_table)
    stam.initialize()
    return stam


def test_multi_table_available_years_union(manager, stam_config_multi_table):
    stam = _multi_table_stam(manager, stam_config_multi_table)
    years = stam.matcher.tabular_source.available_years()
    # crop_yield.csv (2020, 2021) ∪ icrisat_wide.csv (2019, 2020).
    assert 2019 in years and 2020 in years and 2021 in years
    assert max(years) == 2021


def test_multi_table_icrisat_district_fallback(manager, stam_config_multi_table):
    # Point inside the DK district polygon but outside any village/taluk:
    # table 1 (village-level) cannot answer, table 2 (ICRISAT-style) does at
    # district level and its dominant crop (Rice > Cotton by area) wins.
    stam = _multi_table_stam(manager, stam_config_multi_table)
    obs = stam.build_observation(74.86, 13.15, year=2020, season="Kharif")
    assert obs.location.dataset_location_name == "DK"
    assert obs.tabular.matched_level == "district"
    assert obs.tabular.source_path.endswith("icrisat_wide.csv")
    assert obs.crop == "Rice"
    assert obs.yield_value == 4000.0
    assert obs.quality.passed is True


def test_multi_table_no_coverage_anywhere(manager, stam_config_multi_table):
    # Neither table has a row for this district-year: no fabricated fallback,
    # ST-Q-TAB-001 rejects the observation instead of a wrong "match".
    stam = _multi_table_stam(manager, stam_config_multi_table)
    obs = stam.build_observation(74.86, 13.15, year=2015, season="Kharif")
    assert obs.tabular.matched_level == "none"
    assert any(i.code == "ST-Q-TAB-001" for i in obs.quality.issues)


def test_multi_table_first_table_wins(manager, stam_config_multi_table):
    # Village A is in table 1 (crop_yield.csv @2020 Kharif) and also in the
    # icrisat_wide.csv district row: table 1 must win.
    stam = _multi_table_stam(manager, stam_config_multi_table)
    obs = stam.build_observation(74.802, 13.098, year=2020, season="Kharif")
    assert obs.tabular.matched_level == "village"
    assert obs.tabular.source_path.endswith("crop_yield.csv")
    assert obs.crop == "Rice"
    assert obs.yield_value == 5200.0


# --------------------------------------------------------------------------- #
# District / place-name alias (boundary spelling -> data_season Location)
# --------------------------------------------------------------------------- #


def test_district_to_csv():
    # Reverse direction of the CSV->boundary alias table.
    assert district_to_csv("Dakshina Kannada") == "Mangalore"
    assert district_to_csv("Kalaburgi") == "Gulbarga"
    assert district_to_csv("Kalaburagi") == "Gulbarga"
    assert district_to_csv("Bengaluru (Urban)") == "Bangalore"
    assert district_to_csv("Bengaluru (Rural)") == "Bangalore"
    assert district_to_csv("kodagu") is None
    assert district_to_csv("Hassan") is None
    assert district_to_csv(None) is None


def test_district_to_csv_round_trip():
    # For every aliased district the two directions are inverses.
    for csv_name, boundary in [
        ("Mangalore", "Dakshina Kannada"),
        ("Bangalore", "Bengaluru"),
        ("Chikmangaluru", "Chikkamagaluru"),
        ("Davangere", "Davanagere"),
        ("Gulbarga", "Kalaburgi"),
    ]:
        assert normalize_name(csv_name) == boundary
        assert district_to_csv(boundary) == csv_name


def test_district_alias_dakshina_kannada_to_mangalore(alias_stam):
    # A point whose nearest location is the "Dakshina Kannada" district
    # centroid (no village/taluk polygon) must fall back to the data_season
    # "Mangalore" row via the district alias, not fail ST-Q-TAB-001.
    for year in (2018, 2019):
        obs = alias_stam.build_observation(75.0, 12.9, year=year, season="Kharif")
        assert obs.location.dataset_location_name == "Dakshina Kannada"
        assert obs.location.admin.district == "Dakshina Kannada"
        assert obs.tabular.matched_level == "village"
        assert obs.crop == "Coconut"
        assert obs.provenance["tabular_village"] == "Mangalore"


def test_district_alias_kalaburgi_to_gulbarga(alias_stam):
    obs = alias_stam.build_observation(76.95, 17.0, year=2018, season="Kharif")
    assert obs.location.dataset_location_name == "Kalaburgi"
    assert obs.tabular.matched_level == "village"
    assert obs.crop == "Tur"
    assert obs.provenance["tabular_village"] == "Gulbarga"


def test_district_alias_bengaluru_urban_to_bangalore(alias_stam):
    # "Bengaluru (Urban)" -> undivided "Bangalore" row (parenthetical
    # qualifiers are stripped before the alias lookup).
    obs = alias_stam.build_observation(77.55, 13.0, year=2018, season="Kharif")
    assert obs.location.dataset_location_name == "Bengaluru (Urban)"
    assert obs.tabular.matched_level == "village"
    assert obs.crop == "Ragi"
    assert obs.provenance["tabular_village"] == "Bangalore"


def test_district_alias_madikeri_via_taluk(alias_stam):
    # Kodagu has no district alias, so the taluk name "Madikeri" is kept and
    # matches the data_season "Madikeri" row unchanged.
    obs = alias_stam.build_observation(75.55, 12.25, year=2018, season="Kharif")
    assert obs.location.admin.taluk == "Madikeri"
    assert obs.location.admin.district == "Kodagu"
    assert obs.tabular.matched_level == "village"
    assert obs.crop == "Coffee"
    assert obs.provenance["tabular_village"] == "Madikeri"


def test_district_alias_no_coverage_clean_failure(alias_stam):
    # Belgaum has no district alias and no data_season row: ST-Q-TAB-001 is
    # raised as a quality issue (never an exception).
    obs = alias_stam.build_observation(74.65, 16.0, year=2018, season="Kharif")
    assert obs.location.dataset_location_name == "Belgaum"
    assert obs.tabular.matched_level == "none"
    assert any(i.code == "ST-Q-TAB-001" for i in obs.quality.issues)


def test_kasaragodu_resolves_cleanly(alias_stam):
    # Manual point outside the boundary files: no district alias is applied,
    # no fake boundary entry, and the data_season "Kasaragodu" row matches.
    obs = alias_stam.build_observation(74.99, 12.49, year=2018, season="Kharif")
    assert obs.location.dataset_location_name == "Kasaragodu"
    assert obs.location.admin is None
    assert obs.tabular.matched_level == "village"
    assert obs.crop == "Coconut"
