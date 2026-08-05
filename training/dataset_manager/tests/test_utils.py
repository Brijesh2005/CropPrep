"""Unit tests for shared utility helpers."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from training.dataset_manager.models import IndexType, Resolution
from training.dataset_manager.utils import (
    classify_index_type,
    classify_index_type_from_path,
    classify_resolution,
    classify_resolution_from_path,
    count_lines_fast,
    extract_year_from_path,
    human_size,
    is_geotiff_bytes,
    parse_observation_date,
    run_parallel,
    sha256_file,
    tree_signature,
)


def test_sha256_file(tmp_path: Path):
    file = tmp_path / "a.bin"
    file.write_bytes(b"hello world" * 1000)
    digest = sha256_file(file)
    assert len(digest) == 64
    assert digest == sha256_file(file)  # deterministic


def test_count_lines_fast(tmp_path: Path):
    file = tmp_path / "lines.txt"
    file.write_bytes(b"a\nb\nc\n" * 100)
    assert count_lines_fast(file) == 300


def test_human_size():
    assert human_size(0) == "0 B"
    assert human_size(1024) == "1.0 KiB"
    assert "MiB" in human_size(5 * 1024 * 1024)


def test_tiff_magic(tmp_path: Path):
    real = tmp_path / "img.tif"
    real.write_bytes(b"II*\x00\x0f\x00\x00\x00")
    assert is_geotiff_bytes(real)
    bogus = tmp_path / "bogus.tif"
    bogus.write_bytes(b"this is not a tiff")
    assert not is_geotiff_bytes(bogus)


def test_classify_index_type():
    assert classify_index_type("NDVI_2020") is IndexType.NDVI
    assert classify_index_type("EVI_2020") is IndexType.EVI
    assert classify_index_type("foo.tif") is IndexType.NONE
    assert (
        classify_index_type_from_path(Path("x/2019_images/NDVI/file.tif"))
        is IndexType.NDVI
    )


def test_classify_resolution():
    assert classify_resolution("R10m") is Resolution.R10M
    assert classify_resolution("R20m") is Resolution.R20M
    assert classify_resolution("plain") is Resolution.UNKNOWN
    assert (
        classify_resolution_from_path(Path("2019_images/R10m/NDVI.tif"))
        is Resolution.R10M
    )


def test_extract_year_and_date():
    assert extract_year_from_path("2019_images/R10m/NDVI.tif") == 2019
    assert extract_year_from_path("no_year_here") is None
    assert parse_observation_date("S2_NDVI_2019_0701.tif") == date(2019, 7, 1)
    assert parse_observation_date("2019-07-01.tif") == date(2019, 7, 1)
    assert parse_observation_date("nothing.tif") is None


def test_run_parallel_preserves_order():
    results = run_parallel(list(range(8)), lambda x: x * 2, workers=4)
    assert results == [x * 2 for x in range(8)]


def test_run_parallel_captures_exceptions():
    def boom(x):
        if x == 3:
            raise ValueError("bad")
        return x

    results = run_parallel(list(range(5)), boom)
    assert results[3] is not None and isinstance(results[3], ValueError)


def test_tree_signature(tmp_path: Path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "f1").write_bytes(b"x" * 10)
    (tmp_path / "f2").write_bytes(b"y" * 20)
    signature = tree_signature(tmp_path)
    assert signature[0] == 2
    assert signature[1] == 30
