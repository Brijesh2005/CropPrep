"""Tests for the CSV loader (discovery, schema, missing values, statistics)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from training.dataset_manager.csv_loader import PandasCSVLoader
from training.dataset_manager.exceptions import CorruptedDatasetError, UnsupportedFormatError


def _write_csv(tmp_path: Path, content: str, name: str = "data.csv") -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    return _write_csv(
        tmp_path,
        "village,crop,yield_kg,rainfall_mm\n"
        "Moodabidri,Rice,5200,2100\n"
        "Bantwal,Rice,5400,2050\n"
        "Sullia,Coconut,3100,\n"
        "Belthangady,Rice,,2300\n",
    )


def test_discover_recursive(tmp_path: Path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "b").mkdir()
    _write_csv(tmp_path, "x,y\n1,2\n", "top.csv")
    _write_csv(tmp_path / "a", "x,y\n1,2\n", "nested.csv")
    _write_csv(tmp_path / "a" / "b", "x,y\n1,2\n", "deep.csv")
    _write_csv(tmp_path, "not really csv", "ignore.log")
    found = PandasCSVLoader().discover(tmp_path)
    assert len(found) == 3
    assert all(p.suffix == ".csv" for p in found)


def test_profile_schema(sample_csv: Path):
    profile = PandasCSVLoader().profile(sample_csv)
    assert profile.column_count == 4
    assert profile.row_count == 4
    assert profile.columns == ["village", "crop", "yield_kg", "rainfall_mm"]
    # Columns containing missing values are promoted to float by pandas.
    assert profile.dtypes["yield_kg"].startswith("float")
    assert profile.total_missing == 2  # one None in yield_kg, one in rainfall_mm
    assert profile.missing_values["rainfall_mm"] == 1


def test_profile_statistics(sample_csv: Path):
    profile = PandasCSVLoader().profile(sample_csv)
    stats = profile.extra["statistics"]
    assert "yield_kg" in stats
    assert round(stats["yield_kg"]["mean"]) == 4567  # (5200+5400+3100)/3


def test_preview(sample_csv: Path):
    preview = PandasCSVLoader().preview(sample_csv, n_rows=2)
    assert isinstance(preview, pd.DataFrame)
    assert len(preview) == 2


def test_load_chunked(sample_csv: Path):
    iterator = PandasCSVLoader().load(sample_csv, chunksize=2)
    chunks = list(iterator)
    assert len(chunks) == 2
    assert sum(len(c) for c in chunks) == 4


def test_empty_csv_is_corrupted(tmp_path: Path):
    empty = _write_csv(tmp_path, "", "empty.csv")
    with pytest.raises(CorruptedDatasetError):
        PandasCSVLoader().profile(empty)


def test_unsupported_extension(tmp_path: Path):
    path = tmp_path / "data.xyz"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(UnsupportedFormatError):
        PandasCSVLoader().load(path)


def test_encoding_detection(tmp_path: Path):
    path = tmp_path / "latin.csv"
    path.write_bytes("name,v\nJosé,1\n".encode("latin-1"))
    loader = PandasCSVLoader()
    assert loader.guess_encoding(path) in {"latin-1", "utf-8"}
    df = loader.load(path)
    assert df.iloc[0]["name"] == "José"


def test_detect_missing_values(sample_csv: Path):
    missing = PandasCSVLoader().detect_missing_values(sample_csv)
    assert missing["rainfall_mm"] == 1
    assert missing["yield_kg"] == 1
