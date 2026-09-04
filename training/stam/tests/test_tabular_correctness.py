"""Regression tests for the correctness-first tabular matcher.

These exercise the no-fabrication contract: a location that matches no
genuine name must be rejected (``None``) rather than answered with the first
row of the year/season subset, tables are tried in order
(reject-and-try-next-source), and candidates are selected by a deterministic
score rather than first-hits-wins.
"""

from __future__ import annotations

import pandas as pd

from training.stam.config import TabularTableConfig
from training.stam.matcher import (
    DatasetManagerTabularSource,
    _confidence_from_score,
    _name_quality,
)

# --------------------------------------------------------------------------- #
# Confidence mapping
# --------------------------------------------------------------------------- #


def test_confidence_high_at_full_score():
    assert _confidence_from_score(1.0) == "high"


def test_confidence_medium_at_taluk_exact():
    # taluk exact: 0.75 * 1.0 = 0.75
    assert _confidence_from_score(0.75) == "medium"


def test_confidence_low_for_weak_matches():
    assert _confidence_from_score(0.20) == "low"


def test_confidence_none_for_zero():
    assert _confidence_from_score(0.0) == "none"


# --------------------------------------------------------------------------- #
# Name quality classifiers
# --------------------------------------------------------------------------- #


def test_name_quality_exact():
    assert _name_quality("Mangalore", "Mangalore") == "exact"


def test_name_quality_alternate_handles_case_and_space():
    assert _name_quality("Dakshina Kannada", "dakshina kannada") == "exact"


def test_name_quality_alternate_slash():
    assert _name_quality("Gulbarga / Kalaburagi", "Kalaburagi") in ("alternate", "exact")


def test_name_quality_contains_substring():
    assert _name_quality("Bengaluru (Urban)", "Bengaluru") == "contains"


# --------------------------------------------------------------------------- #
# Candidate scoring (deterministic, best-wins)
# --------------------------------------------------------------------------- #


def _frame(values):
    return pd.DataFrame({"Location": values, "Crops": ["Coconut"] * len(values)})


def _source(manager):
    return DatasetManagerTabularSource(manager)


def test_score_candidates_prefers_exact_over_contains(manager):
    subset = _frame(["Bengaluru (Urban)", "Bengaluru", "Raichur"])
    subset["yeilds"] = [50.0, 60.0, 70.0]
    cfg = TabularTableConfig(
        name="t.csv", village_column="Location", crop_column="Crops",
        yield_column="yeilds",
    )
    score, row, level = _source(manager)._score_candidates(
        subset, cfg, "Bengaluru", None, None
    )
    assert level == "village"
    # Best score should come from the exact "Bengaluru" (60) over the
    # contains "Bengaluru (Urban)" (50).
    assert row["yeilds"] == 60.0
    assert score == 1.0


def test_score_candidates_returns_none_when_no_genuine_match(manager):
    subset = _frame(["Raichur", "Belgaum"])
    cfg = TabularTableConfig(name="t.csv", village_column="Location")
    assert _source(manager)._score_candidates(
        subset, cfg, "NowhereVillage", None, None
    ) is None


def test_score_candidates_ranks_village_over_district(manager):
    subset = pd.DataFrame(
        {
            "village": ["A", "Z"],
            "district": ["DK", "DK"],
            "crop": ["Rice", "Coconut"],
            "year": [2020, 2020],
            "season": ["Kharif", "Kharif"],
        }
    )
    cfg = TabularTableConfig(
        name="t.csv", village_column="village", district_column="district",
        crop_column="crop", year_column="year", season_column="season",
        fallback_to_district=True,
    )
    # Query village A -> its row (village level, score 1.0) outranks the
    # district-wide rows for Z (district level, score 0.45).
    score, row, level = _source(manager)._score_candidates(
        subset, cfg, "A", None, "DK"
    )
    assert level == "village"
    assert row["crop"] == "Rice"


# --------------------------------------------------------------------------- #
# Integration: no-fabrication + reject-and-try-next-source
# --------------------------------------------------------------------------- #


def test_load_record_returns_none_when_no_genuine_match(
    manager, tmp_path
):
    """A location with no genuine name anywhere returns None (no fabrication)."""
    table_path = tmp_path / "only.csv"
    pd.DataFrame(
        {
            "Location": ["Mangalore", "Raichur"],
            "Crops": ["Coconut", "Rice"],
            "yeilds": [100.0, 50.0],
            "Year": [2020, 2020],
            "Season": ["Kharif", "Kharif"],
        }
    ).to_csv(table_path, index=False)

    cfg = TabularTableConfig(
        name=table_path.name, village_column="Location", year_column="Year",
        season_column="Season", crop_column="Crops", yield_column="yeilds",
    )
    # Point the Dataset Manager's CSV listing at this single table. We build a
    # real DatasetManagerTabularSource over the real manager but override
    # _load_tables to return only our synthetic table so no real CSV (which is
    # not mounted locally) is required.
    source = DatasetManagerTabularSource(manager, _config_containing(cfg))
    source._frames = {cfg.name: (cfg, table_path, pd.read_csv(table_path))}

    result = source.load_record(
        village="NonexistentVillage", taluk=None, district=None, year=2020,
        season="Kharif",
    )
    assert result is None


def test_load_record_returns_genuine_match_with_provenance(manager, tmp_path):
    table_path = tmp_path / "only.csv"
    pd.DataFrame(
        {
            "Location": ["Mangalore", "Raichur"],
            "Crops": ["Coconut", "Rice"],
            "yeilds": [100.0, 50.0],
            "Year": [2020, 2020],
            "Season": ["Kharif", "Kharif"],
        }
    ).to_csv(table_path, index=False)
    cfg = TabularTableConfig(
        name=table_path.name, village_column="Location", year_column="Year",
        season_column="Season", crop_column="Crops", yield_column="yeilds",
    )
    source = DatasetManagerTabularSource(manager, _config_containing(cfg))
    source._frames = {cfg.name: (cfg, table_path, pd.read_csv(table_path))}

    result = source.load_record(
        village="Mangalore", taluk=None, district=None, year=2020, season="Kharif",
    )
    assert result is not None
    assert result["__matched_level"] == "village"
    assert result["__match_score"] == 1.0
    assert result["__confidence"] == "high"
    assert result["__source_table"] == table_path.name
    assert result["__source_path"] == str(table_path)
    assert result["Crops"] == "Coconut"


def test_load_record_reject_and_try_next_source(manager, tmp_path):
    """First table missing -> falls through to second table's genuine match."""
    t1 = tmp_path / "t1.csv"
    pd.DataFrame(
        {
            "Location": ["Kalaburgi"],
            "Crops": ["Rice"],
            "yeilds": [1.0],
            "Year": [2020],
        }
    ).to_csv(t1, index=False)
    t2 = tmp_path / "t2.csv"
    pd.DataFrame(
        {
            "Dist Name": ["Dakshina Kannada"],
            "RICE AREA (1000 ha)": [100.0],
            "RICE YIELD (Kg per ha)": [4000.0],
            "Year": [2020],
            "State Name": ["Karnataka"],
        }
    ).to_csv(t2, index=False)

    cfg1 = TabularTableConfig(
        name=t1.name, village_column="Location", year_column="Year",
        crop_column="Crops", yield_column="yeilds",
    )
    cfg2 = TabularTableConfig(
        name=t2.name, district_column="Dist Name", year_column="Year",
        state_column="State Name", state_value="Karnataka",
    )
    source = DatasetManagerTabularSource(
        manager, _config_containing([cfg1, cfg2])
    )
    source._frames = {
        cfg1.name: (cfg1, t1, pd.read_csv(t1)),
        cfg2.name: (cfg2, t2, pd.read_csv(t2)),
    }

    result = source.load_record(
        village="Dakshina Kannada", taluk=None, district="Dakshina Kannada",
        year=2020, season="Kharif",
    )
    assert result is not None
    assert result["__source_table"] == t2.name
    assert result["__matched_level"] == "district"


def _config_containing(tables):
    from training.stam.config import TabularConfig

    if isinstance(tables, TabularTableConfig):
        tables = [tables]
    return TabularConfig(tables=tables)
