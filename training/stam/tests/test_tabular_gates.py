"""Tests for the tabular-source auto-approval gates + wire-in helpers."""

from __future__ import annotations

import yaml

import pandas as pd
import pytest

from training.stam.tabular_gates import (
    DEFAULT_NAME_ALIASES,
    add_aliases,
    alias_suggestion,
    append_table,
    gate_a,
    gate_b,
    main,
    review,
    review_source,
)
from training.stam.tabular_profiler import (
    BoundaryVocabulary,
    candidate_table_entry,
    profile_tabular_source,
)

DISTRICTS = [
    "Dakshina Kannada", "Kodagu", "Kalaburgi", "Belgaum", "Bengaluru (Urban)",
    "Davangere", "Hassan", "Mysuru", "Raichur", "Gulbarga", "Bellary",
    "Chikmagalur", "Kasaragod", "Shimoga",
]
TALUKS = ["Madikeri", "Buntwal", "Puttur"]
ALIAS_KEYS = ["bangalore", "mysore", "mangalore", "gulburga"]
ALIAS_VALUES = ["Bengaluru (Urban)", "Mysuru", "Mangalore", "Gulbarga"]


@pytest.fixture
def vocab() -> BoundaryVocabulary:
    return BoundaryVocabulary(
        districts=DISTRICTS,
        taluks=TALUKS,
        alias_keys=ALIAS_KEYS,
        alias_values=ALIAS_VALUES,
    )


def profile_tabular_source_from_frame(
    frame: pd.DataFrame, vocab: BoundaryVocabulary
):
    """Profile an in-memory frame via a throwaway CSV (no shapefiles needed)."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/table.csv"
        frame.to_csv(path, index=False)
        return profile_tabular_source(path, vocab=vocab)


def _profile(frame: pd.DataFrame, vocab: BoundaryVocabulary, tmp_path):
    path = tmp_path / "table.csv"
    frame.to_csv(path, index=False)
    return profile_tabular_source(path, vocab=vocab)


def _narrow_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Location": ["Mangalore", "Kodagu", "Raichur", "Mysuru"],
            "Year": [2018, 2018, 2019, 2019],
            "Season": ["Kharif", "Kharif", "Rabi", "Rabi"],
            "Crops": ["Coconut", "Coffee", "Rice", "Rice"],
            "yeilds": [11.4, 32.0, 54.0, 86.0],
        }
    )


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


# --------------------------------------------------------------------------- #
# Gate A
# --------------------------------------------------------------------------- #


def test_gate_a_passes_narrow_table(vocab, tmp_path):
    review_ = review(_profile(_narrow_frame(), vocab, tmp_path))
    assert all(check.ok for check in review_.gate_a)
    assert review_.approved


def test_gate_a_rejects_wide_format(vocab, tmp_path):
    review_ = review(_profile(_icrisat_frame(), vocab, tmp_path))
    assert review_.profile.wide_format is True
    by_name = {check.name: check for check in review_.gate_a}
    assert by_name["yield"].ok is False
    assert by_name["crop"].ok is False
    assert review_.approved is False


def test_gate_a_rejects_missing_year(vocab, tmp_path):
    frame = pd.DataFrame(
        {
            "Location": ["Mangalore", "Kodagu"],
            "Crops": ["Coconut", "Coffee"],
            "yeilds": [11.4, 32.0],
        }
    )
    review_ = review(_profile(frame, vocab, tmp_path))
    by_name = {check.name: check for check in review_.gate_a}
    assert by_name["year"].ok is False


# --------------------------------------------------------------------------- #
# Gate B
# --------------------------------------------------------------------------- #


def test_gate_b_blocks_unmapped_names(vocab, tmp_path):
    frame = pd.DataFrame(
        {
            "Location": ["Mangalore", "Somewhere Unknown"],
            "Year": [2018, 2019],
            "Crops": ["Coconut", "Coffee"],
            "yeilds": [11.4, 32.0],
        }
    )
    review_ = review(_profile(frame, vocab, tmp_path))
    by_name = {check.name: check for check in review_.gate_b}
    assert by_name["no-unmapped"].ok is False
    assert review_.approved is False


def test_gate_b_blocks_ambiguous_names():
    vocab = BoundaryVocabulary(
        districts=["Bengaluru (Urban)", "Bengaluru (Rural)", "Bengaluru (South)"],
        taluks=[],
    )
    frame = pd.DataFrame(
        {
            "Location": ["Bengaluru (Urban)", "Bengaluru (Rural)"],
            "Year": [2018, 2019],
            "Crops": ["Coconut", "Coffee"],
            "yeilds": [11.4, 32.0],
        }
    )
    profile = profile_tabular_source_from_frame(frame, vocab)
    review_ = review(profile)
    by_name = {check.name: check for check in review_.gate_b}
    assert by_name["no-ambiguous"].ok is False


def test_gate_b_suggests_but_does_not_block_aliases(vocab, tmp_path):
    frame = pd.DataFrame(
        {
            "Location": ["Mangalore", "Bengalur"],
            "Year": [2018, 2019],
            "Crops": ["Coconut", "Coffee"],
            "yeilds": [11.4, 32.0],
        }
    )
    review_ = review(_profile(frame, vocab, tmp_path))
    assert review_.approved is True
    assert [v.name for v in review_.suggested_aliases] == ["Bengalur"]


def test_alias_suggestion_strips_parenthetical():
    from training.stam.tabular_profiler import NameMatch, PlaceNameVerdict

    verdict = PlaceNameVerdict(
        "Bengalur", "alias", 0.9412,
        matches=(NameMatch("Bengaluru (Urban)", 0.9412, "district"),),
    )
    assert alias_suggestion(verdict) == ("Bengalur", "Bengaluru")


# --------------------------------------------------------------------------- #
# append_table (stam.yaml)
# --------------------------------------------------------------------------- #


@pytest.fixture
def stam_yaml(tmp_path):
    path = tmp_path / "stam.yaml"
    path.write_text(
        "tabular:\n"
        "  tables:\n"
        "    - name: a.csv\n"
        "      year_column: Year\n"
        "# next block\n"
        "seasons:\n"
        "  - name: Kharif\n",
        encoding="utf-8",
    )
    return path


def test_append_table_adds_entry(stam_yaml):
    entry = {
        "name": "b.csv",
        "year_column": "Year",
        "crop_column": "Crops",
        "fallback_to_district": False,
    }
    assert append_table(stam_yaml, entry) == "added"
    doc = yaml.safe_load(stam_yaml.read_text(encoding="utf-8"))
    tables = doc["tabular"]["tables"]
    assert [t["name"] for t in tables] == ["a.csv", "b.csv"]
    assert tables[1]["crop_column"] == "Crops"


def test_append_table_deduplicates(stam_yaml):
    entry = {"name": "a.csv", "year_column": "Year"}
    assert append_table(stam_yaml, entry) == "duplicate"
    text = stam_yaml.read_text(encoding="utf-8")
    assert text.count("name: a.csv") == 1


def test_append_table_missing_tables_block(tmp_path):
    path = tmp_path / "stam.yaml"
    path.write_text("tabular:\n  table: single.csv\n", encoding="utf-8")
    assert append_table(path, {"name": "x.csv"}) == "missing-tables"


# --------------------------------------------------------------------------- #
# add_aliases (name_aliases.py)
# --------------------------------------------------------------------------- #


def test_add_aliases_inserts_and_deduplicates(tmp_path):
    path = tmp_path / "name_aliases.py"
    path.write_text(
        "ALIASES: dict[str, str] = {\n"
        '    "Mangalore": "Dakshina Kannada",\n'
        "}\n\n"
        'NO_ALIAS_NEEDED: set[str] = {"Hassan"}\n',
        encoding="utf-8",
    )
    added = add_aliases(
        path,
        {"Chikmangaluru": "Chikkamagaluru", "Mangalore": "Dakshina Kannada"},
    )
    assert added == ["Chikmangaluru"]
    text = path.read_text(encoding="utf-8")
    assert '"Chikmangaluru": "Chikkamagaluru",' in text
    assert text.count('"Mangalore"') == 1


# --------------------------------------------------------------------------- #
# CLI smoke tests (real vocabulary, dry-run only)
# --------------------------------------------------------------------------- #


def _approved_real_csv(tmp_path) -> str:
    frame = pd.DataFrame(
        {
            "Location": ["Dakshina Kannada", "Mysuru", "Raichur", "Kodagu"],
            "Year": [2018, 2018, 2019, 2019],
            "Season": ["Kharif", "Kharif", "Rabi", "Rabi"],
            "Crops": ["Coconut", "Coffee", "Rice", "Rice"],
            "yeilds": [11.4, 32.0, 54.0, 86.0],
        }
    )
    path = tmp_path / "approved.csv"
    frame.to_csv(path, index=False)
    return str(path)


def test_cli_add_dry_run(tmp_path, capsys):
    path = _approved_real_csv(tmp_path)
    rc = main(["add", path, "--dry-run"])
    assert rc == 0
    assert "Gate A" in capsys.readouterr().out


def test_cli_add_blocks_unmapped(tmp_path, capsys):
    path = tmp_path / "bad.csv"
    pd.DataFrame(
        {
            "Location": ["Nowhere At All"],
            "Year": [2018],
            "Crops": ["Coconut"],
            "yeilds": [11.4],
        }
    ).to_csv(path, index=False)
    rc = main(["add", str(path), "--dry-run"])
    assert rc == 1
    assert "NOT auto-approved" in capsys.readouterr().out


def test_cli_list(tmp_path, capsys):
    _approved_real_csv(tmp_path)
    rc = main(["list", "--tabular-dir", str(tmp_path)])
    assert rc == 0
    assert "approved.csv" in capsys.readouterr().out


def test_candidate_entry_keys_fit_config():
    """Committed entries must only use keys TabularTableConfig accepts."""
    from training.stam.config import TabularTableConfig

    vocab = BoundaryVocabulary(districts=DISTRICTS, taluks=TALUKS,
                               alias_keys=ALIAS_KEYS, alias_values=ALIAS_VALUES)
    narrow = pd.DataFrame(
        {
            "Location": ["Mangalore", "Kodagu", "Mysuru"],
            "Year": [2018, 2019, 2020],
            "Crops": ["Coconut", "Coffee", "Rice"],
            "yeilds": [11.4, 32.0, 54.0],
        }
    )
    profile = profile_tabular_source_from_frame(narrow, vocab)
    entry = candidate_table_entry(profile)
    assert entry["name"] == "table.csv"
    cfg = TabularTableConfig(**entry)
    assert cfg.name == "table.csv"
    assert cfg.crop_column == "Crops"
