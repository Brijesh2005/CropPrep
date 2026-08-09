"""Tests for the tabular-source profiler (field inference + Gate B verdicts).

Uses a small in-memory ``BoundaryVocabulary`` so no real KGIS shapefiles are
touched; ``profile_tabular_source`` receives it via the ``vocab`` injection
point.
"""

from __future__ import annotations

import pandas as pd
import pytest

from training.stam.tabular_profiler import (
    BoundaryVocabulary,
    SourceProfile,
    build_default_vocabulary,
    candidate_table_entry,
    clean_name,
    infer_crop,
    infer_location,
    infer_season,
    infer_state,
    infer_year,
    key_name,
    profile_tabular_source,
)

DISTRICTS = [
    "Dakshina Kannada", "Kodagu", "Kalaburgi", "Belgaum", "Bengaluru (Urban)",
    "Davangere", "Hassan", "Mysuru", "Raichur", "Gulbarga", "Bellary",
    "Chikmagalur", "Kasaragod", "Shimoga",
]
TALUKS = ["Madikeri", "Buntwal", "Puttur"]
ALIAS_KEYS = ["bangalore", "mysore", "mangalore", "gulburga", "shimoga"]
ALIAS_VALUES = ["Bengaluru (Urban)", "Mysuru", "Mangalore", "Gulbarga", "Shimoga"]


@pytest.fixture
def vocab() -> BoundaryVocabulary:
    return BoundaryVocabulary(
        districts=DISTRICTS,
        taluks=TALUKS,
        alias_keys=ALIAS_KEYS,
        alias_values=ALIAS_VALUES,
    )


def _profile(frame: pd.DataFrame, vocab: BoundaryVocabulary, tmp_path) -> SourceProfile:
    path = tmp_path / "table.csv"
    frame.to_csv(path, index=False)
    return profile_tabular_source(path, vocab=vocab)


# --------------------------------------------------------------------------- #
# Naming helpers
# --------------------------------------------------------------------------- #


def test_clean_name_collapses_whitespace():
    assert clean_name("  Dakshina   Kannada  ") == "Dakshina Kannada"


def test_key_name_strips_parenthetical_and_lowercases():
    assert key_name("Bengaluru (Urban)") == "bengaluru"


# --------------------------------------------------------------------------- #
# Narrow-format source (data_season style) -> all gates satisfied
# --------------------------------------------------------------------------- #


def test_narrow_table_full_profile(vocab, tmp_path):
    frame = pd.DataFrame(
        {
            "Location": ["Mangalore", "Kodagu", "Raichur", "Mysuru"],
            "Year": [2018, 2018, 2019, 2019],
            "Season": ["Kharif", "Kharif", "Rabi", "Rabi"],
            "Crops": ["Coconut", "Coffee", "Rice", "Rice"],
            "yeilds": [11.4, 32.0, 54.0, 86.0],
            "Area": [100, 200, 300, 400],
            "Rainfall": [2900, 2400, 900, 850],
        }
    )
    p = _profile(frame, vocab, tmp_path)
    assert p.wide_format is False
    assert (p.location.column, p.location.confidence) == ("Location", 1.0)
    assert p.year.column == "Year"
    assert p.season.column == "Season"
    assert p.crop.column == "Crops"
    assert p.yield_.column == "yeilds"
    assert p.yield_.method == "name-based-match"
    assert p.state.column is None
    assert p.feature_columns == ["Area", "Rainfall"]
    assert set((v.name, v.status) for v in p.place_verdicts) >= {
        ("Mangalore", "exact"),
        ("Kodagu", "exact"),
    }
    cfg = p.table_config
    assert cfg["village_column"] == "Location"
    assert cfg["season_column"] == "Season"
    assert cfg["yield_column"] == "yeilds"
    assert cfg["fallback_to_district"] is False


def test_wide_format_not_detected_for_narrow(vocab, tmp_path):
    frame = pd.DataFrame(
        {
            "Location": ["Mangalore", "Kodagu"],
            "Crops": ["Coconut", "Coffee"],
            "yeilds": [11.4, 32.0],
        }
    )
    p = _profile(frame, vocab, tmp_path)
    assert p.wide_format is False


# --------------------------------------------------------------------------- #
# Wide-format ICRISAT-style source + state restriction
# --------------------------------------------------------------------------- #


def _icrisat_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "State Name": ["Karnataka"] * 3 + ["Chhattisgarh"] * 3,
            "Dist Name": ["Belgaum", "Hassan", "Mysuru",
                          "Raipur", "Durg", "Bastar"],
            "Year": [2020, 2020, 2020, 2020, 2020, 2020],
            "RICE AREA (1000 ha)": [100.0, 110.0, 120.0, 200.0, 210.0, 220.0],
            "RICE PRODUCTION (1000 tons)": [4000.0, 4400.0, 4800.0, 8000.0, 8400.0, 8800.0],
            "RICE YIELD (Kg per ha)": [4000.0, 4000.0, 4000.0, 4000.0, 4000.0, 4000.0],
            "WHEAT AREA (1000 ha)": [50.0, 40.0, 60.0, 90.0, 80.0, 70.0],
            "WHEAT PRODUCTION (1000 tons)": [300.0, 240.0, 360.0, 540.0, 480.0, 420.0],
            "WHEAT YIELD (Kg per ha)": [6000.0, 6000.0, 6000.0, 6000.0, 6000.0, 6000.0],
        }
    )


def test_wide_format_detected_and_state_restricted(vocab, tmp_path):
    p = _profile(_icrisat_frame(), vocab, tmp_path)
    assert p.wide_format is True
    assert p.yield_.method == "wide-format"
    assert p.crop.column is None
    assert p.state.column == "State Name"
    assert p.state_value == "Karnataka"
    assert p.location.column == "Dist Name"
    assert p.district.column == "Dist Name"
    assert p.taluk.column is None
    assert "(state=Karnataka)" in p.location.note
    cfg = p.table_config
    assert cfg["district_column"] == "Dist Name"
    assert cfg["fallback_to_district"] is True
    assert cfg["state_column"] == "State Name"
    assert cfg["state_value"] == "Karnataka"


def test_state_restriction_prefers_high_coverage_state(vocab):
    """Karnataka rows cover the boundary vocabulary; Chhattisgarh rows do not,
    so the profiler must restrict the location column to Karnataka."""
    location, state_inf, _, state_value = infer_location(
        _icrisat_frame(), vocab, set(), infer_state(_icrisat_frame(), vocab, set())
    )
    assert state_value == "Karnataka"
    assert location.column == "Dist Name"
    assert location.confidence == 1.0


# --------------------------------------------------------------------------- #
# Per-field inference
# --------------------------------------------------------------------------- #


def test_infer_state_ignores_non_state_columns(vocab):
    frame = pd.DataFrame(
        {
            "Location": ["Mangalore", "Kodagu", "Raichur"],
            "Soil type": ["Alluvial", "Red", "Black"],
        }
    )
    assert infer_state(frame, vocab, set()).column is None


def test_infer_year_ambiguous_flag(vocab):
    frame = pd.DataFrame(
        {
            "Year": [2018, 2019, 2020],
            "Census": [2001, 2011, 2021],
        }
    )
    inf = infer_year(frame)
    assert inf.column is None
    assert inf.method == "ambiguous-year"


def test_infer_year_single(vocab):
    frame = pd.DataFrame({"Year": [2018, 2019, 2020]})
    inf = infer_year(frame)
    assert inf.column == "Year"
    assert inf.confidence == 1.0


def test_infer_season_overlap(vocab):
    frame = pd.DataFrame(
        {
            "Season": ["Kharif", "Rabi", "Zaid"],
            "Notes": ["x", "y", "z"],
        }
    )
    inf = infer_season(frame, set())
    assert inf.column == "Season"
    assert inf.confidence == 1.0


def test_infer_crop_overlap(vocab):
    frame = pd.DataFrame(
        {
            "Crops": ["Coconut", "Coffee", "Rice", "Tur"],
            "Location": ["Mangalore", "Kodagu", "Raichur", "Mysuru"],
        }
    )
    inf = infer_crop(frame, {"Location"})
    assert inf.column == "Crops"


# --------------------------------------------------------------------------- #
# Gate-B verdicts
# --------------------------------------------------------------------------- #


def test_classify_exact(vocab):
    verdict = vocab.classify("Kodagu")
    assert verdict.status == "exact"
    assert verdict.score == 1.0


def test_classify_alias_existing(vocab):
    verdict = vocab.classify("Bangalore")
    assert verdict.status == "alias_existing"
    assert verdict.score == 1.0


def test_classify_unmapped_below_similarity(vocab):
    verdict = vocab.classify("Bastar")
    assert verdict.status == "unmapped"
    assert verdict.score < 0.92


def test_classify_ambiguous_same_key(vocab):
    extra = BoundaryVocabulary(
        districts=["Bengaluru (Urban)", "Bengaluru (Rural)", "Bengaluru (South)"],
        taluks=[],
    )
    verdict = extra.classify("Bengaluru (Urban)")
    assert verdict.status == "ambiguous"
    assert len(verdict.matches) >= 2


# --------------------------------------------------------------------------- #
# candidate_table_entry
# --------------------------------------------------------------------------- #


def test_candidate_entry_narrow(vocab, tmp_path):
    frame = pd.DataFrame(
        {
            "Location": ["Mangalore", "Kodagu"],
            "Year": [2018, 2019],
            "Season": ["Kharif", "Kharif"],
            "Crops": ["Coconut", "Coffee"],
            "yeilds": [11.4, 32.0],
        }
    )
    p = _profile(frame, vocab, tmp_path)
    entry = candidate_table_entry(p)
    assert entry["name"] == "table.csv"
    assert entry["yield_column"] == "yeilds"


def test_candidate_entry_wide(vocab, tmp_path):
    p = _profile(_icrisat_frame(), vocab, tmp_path)
    entry = candidate_table_entry(p)
    assert entry["district_column"] == "Dist Name"
    assert entry["state_value"] == "Karnataka"
    assert entry["yield_column"] is None


# --------------------------------------------------------------------------- #
# Default vocabulary builds from the real repo resources
# --------------------------------------------------------------------------- #


def test_build_default_vocabulary_uses_real_gis():
    vocab = build_default_vocabulary()
    assert len(vocab.districts) > 20
    assert len(vocab.taluks) > 20
    # A well-known KGIS spelling resolves with full confidence (it is a real
    # boundary name, even when it also appears as an alias value).
    assert vocab.best_match("Dakshina Kannada").score == 1.0
    assert vocab.best_match("Dakshina Kannada").kind != "none"
