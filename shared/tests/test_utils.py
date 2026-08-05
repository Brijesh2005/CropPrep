"""Tests for shared utilities (hash / io / naming / env / yaml / json)."""

from __future__ import annotations

import json

import pytest

from shared.enums import IndexType, Resolution
from shared.utils import (
    classify_index_type,
    classify_resolution,
    count_lines_fast,
    env_map_of,
    extract_year_from_path,
    human_size,
    is_csv_path,
    is_geotiff_bytes,
    is_geotiff_path,
    parse_observation_date,
    parse_json,
    sha256_file,
    walk_files,
    write_json,
    write_yaml,
    yaml_safe,
)


def test_sha256_file(tmp_path) -> None:
    f = tmp_path / "data.bin"
    f.write_bytes(b"hello world" * 1000)
    digest = sha256_file(f)
    assert len(digest) == 64
    assert digest == sha256_file(str(f))


def test_count_lines_fast(tmp_path) -> None:
    f = tmp_path / "rows.csv"
    f.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    assert count_lines_fast(f) == 3


def test_human_size() -> None:
    assert human_size(0) == "0 B"
    assert human_size(1024) == "1.0 KiB"
    assert human_size(5 * 1024 * 1024) == "5.0 MiB"


def test_geotiff_detection(tmp_path) -> None:
    tiff = tmp_path / "x.tif"
    tiff.write_bytes(b"II*\x00" + b"\x00" * 16)
    assert is_geotiff_bytes(tiff)
    assert is_geotiff_path(tiff)

    csv = tmp_path / "x.csv"
    csv.write_text("a\n1\n", encoding="utf-8")
    assert is_csv_path(csv)
    assert not is_geotiff_path(csv)


def test_classify_index_type() -> None:
    assert classify_index_type("NDVI_2020") is IndexType.NDVI
    assert classify_index_type("evi_10m") is IndexType.EVI
    assert classify_index_type("other") is IndexType.NONE


def test_classify_resolution() -> None:
    assert classify_resolution("S2A_R10m") is Resolution.R10M
    assert classify_resolution("S2A_R20m") is Resolution.R20M
    assert classify_resolution("plain") is Resolution.UNKNOWN


def test_extract_year() -> None:
    assert extract_year_from_path("raw/2020/NDVI.tif") == 2020
    assert extract_year_from_path("no/year/here") is None


def test_parse_observation_date() -> None:
    assert parse_observation_date("S2A_2021-06-15_NDVI.tif") is not None
    assert parse_observation_date("no-date.tif") is None


def test_yaml_safe_converts_paths(tmp_path) -> None:
    from pathlib import Path

    out = yaml_safe({"p": Path("a/b"), "items": [1, 2]})
    assert out["p"] == str(Path("a/b"))
    assert out["items"] == [1, 2]


def test_write_and_read_yaml(tmp_path) -> None:
    path = write_yaml(tmp_path / "c.yaml", {"n": 3, "p": tmp_path})
    from shared.utils import load_yaml

    data = load_yaml(path)
    assert data["n"] == 3
    assert data["p"] == str(tmp_path)


def test_write_json_and_default_encoder(tmp_path) -> None:
    from shared.utils import read_json

    path = write_json(tmp_path / "d.json", {"when": "2021-01-01", "v": 1})
    data = read_json(path)
    assert data["v"] == 1


def test_parse_json_helpers() -> None:
    assert parse_json('{"a": 1}') == {"a": 1}
    assert parse_json("plain") == "plain"


def test_env_map_of() -> None:
    env = {"DM_ROOT": "x", "DM_SCAN__WORKERS": "4", "OTHER": "y"}
    assert env_map_of("DM_", env) == {"ROOT": "x", "SCAN__WORKERS": "4"}


def test_walk_files_excludes_state(tmp_path) -> None:
    (tmp_path / ".cropfusion").mkdir()
    (tmp_path / ".cropfusion" / "a.txt").write_text("x", encoding="utf-8")
    (tmp_path / "keep.txt").write_text("y", encoding="utf-8")
    names = [p.name for p in walk_files(tmp_path)]
    assert names == ["keep.txt"]
