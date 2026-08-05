"""Tests for the image loader (rasterio + lightweight TIFF parser)."""

from __future__ import annotations

from pathlib import Path

import pytest

from training.dataset_manager.exceptions import CorruptedDatasetError, UnsupportedFormatError
from training.dataset_manager.image_loader import RasterioImageLoader
from training.dataset_manager.models import IndexType, Resolution
from training.dataset_manager.tests.helpers import make_tiff


def test_read_metadata_via_rasterio(tmp_path: Path):
    path = make_tiff(tmp_path / "2019_images" / "R10m" / "S2_NDVI_20190701.tif")
    loader = RasterioImageLoader()
    meta = loader.read_metadata(path)
    assert meta.width == 20 and meta.height == 20
    assert meta.bands == 1
    assert meta.dtype == "float32"
    assert meta.crs == "EPSG:32643"
    assert meta.pixel_size is not None
    assert meta.bounds is not None
    assert meta.index_type is IndexType.NDVI
    assert meta.resolution is Resolution.R10M
    assert meta.year == 2019
    assert meta.observation_date is not None


def test_light_parser_matches_rasterio(tmp_path: Path):
    path = make_tiff(tmp_path / "NDVI" / "S2_EVI_2020.tif")
    loader = RasterioImageLoader()
    light = loader.read_metadata(path, prefer_light=True)
    assert light.width == 20 and light.height == 20
    assert light.bands == 1
    assert light.dtype == "float32"
    assert light.pixel_size is not None
    assert light.bounds is not None
    # CRS is not resolved by the light parser (no geokey parsing).
    assert light.crs is None
    # Index detection prefers the filename ("S2_EVI_...") over the folder.
    assert light.index_type is IndexType.EVI


def test_corrupted_tiff_raises(tmp_path: Path):
    bogus = tmp_path / "bad.tif"
    bogus.write_bytes(b"this is definitely not a tiff")
    with pytest.raises(CorruptedDatasetError):
        RasterioImageLoader().read_metadata(bogus)


def test_unsupported_extension(tmp_path: Path):
    path = tmp_path / "data.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n")
    with pytest.raises(UnsupportedFormatError):
        RasterioImageLoader().read_metadata(path)


def test_preview_returns_stats(tmp_path: Path):
    path = make_tiff(tmp_path / "NDVI" / "S2_NDVI_2020.tif", fill=0.5)
    preview = RasterioImageLoader().preview(path)
    assert preview["width"] == 20
    assert "sample_stats" in preview
    stats = preview["sample_stats"]
    assert round(stats["mean"], 1) == 0.5


def test_read_window(tmp_path: Path):
    path = make_tiff(tmp_path / "NDVI" / "S2_EVI_2020.tif")
    window = RasterioImageLoader().read_window(path, window=(0, 0, 5, 5), band=1)
    assert window.shape == (5, 5)


def test_tiff_variants(tmp_path: Path):
    # uint16 and int16 rasters resolve to the correct dtype strings.
    u16 = make_tiff(tmp_path / "NDVI" / "u16.tif", dtype="uint16", fill=1)
    i16 = make_tiff(tmp_path / "EVI" / "i16.tif", dtype="int16", fill=1)
    loader = RasterioImageLoader()
    assert loader.read_metadata(u16).dtype == "uint16"
    assert loader.read_metadata(i16).dtype == "int16"
