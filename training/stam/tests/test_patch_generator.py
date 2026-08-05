"""Tests for the spatial patch generator (edge correction, padding)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from training.dataset_manager.tests.helpers import make_tiff
from training.stam.exceptions import PatchOutOfBoundsError
from training.stam.observation import ImageRecordRef
from training.stam.patch_generator import SpatialPatchGenerator
from training.stam.interfaces import ImageMetadataSource, ImageReader

# Pixel size and origin used by make_tiff: 20x20 raster at (74.8, 13.0).
_ORIGIN = (74.8, 13.0)
_PIXEL = 0.0001


class _FakeReader(ImageReader):
    """Reads windowed arrays from plain ndarray rasters (via rasterio)."""

    def __init__(self) -> None:
        self.loaded_windows = []

    def read_window(self, path, window, band=1):
        import rasterio

        self.loaded_windows.append(window)
        with rasterio.open(path) as src:
            from rasterio.windows import Window

            rwin = Window(window[1], window[0], window[3], window[2])
            return src.read(band, window=rwin)

    def read_metadata(self, path):
        import rasterio

        with rasterio.open(path) as src:
            return {
                "width": src.width,
                "height": src.height,
                "crs": src.crs.to_string() if src.crs else None,
                "bounds": (float(src.bounds.left), float(src.bounds.bottom),
                           float(src.bounds.right), float(src.bounds.top)),
                "pixel_size": (float(src.res[0]), float(src.res[1])),
            }


class _FakeMetadataSource(ImageMetadataSource):
    def __init__(self, path: Path) -> None:
        self.path = path

    def query_images(self, **kwargs):
        return []

    def image_metadata(self, path: str) -> ImageRecordRef:
        import rasterio

        with rasterio.open(path) as src:
            bounds = src.bounds
            return ImageRecordRef(
                path=str(path),
                relative_path=str(Path(path).name),
                index_type="NDVI",
                resolution="R10m",
                observation_date=None,
                crs=src.crs.to_string() if src.crs else None,
                pixel_size=(float(src.res[0]), float(src.res[1])),
                bounds=(float(bounds.left), float(bounds.bottom),
                        float(bounds.right), float(bounds.top)),
                width=src.width,
                height=src.height,
            )


@pytest.fixture
def raster(tmp_path: Path):
    # EPSG:4326 + degree transform keeps the point->pixel math exact.
    path = make_tiff(
        tmp_path / "S2_NDVI_2020.tif", width=20, height=20, fill=0.5, crs="EPSG:4326"
    )
    return path


@pytest.fixture
def generator(raster):
    return SpatialPatchGenerator(_FakeReader(), _FakeMetadataSource(raster), default_size=8)


def test_center_patch_full_valid(raster, generator):
    # Center of a 20x20 raster at origin (74.8, 13.0), pixel 0.0001.
    patch = generator.get_patch(str(raster), 74.801, 12.999, size=8)
    assert patch.shape == (8, 8)
    assert patch.valid_ratio == 1.0
    assert patch.padded is False


def test_edge_patch_pads_and_masks(raster, generator):
    # Request a 16x16 patch at the top-left corner -> padded.
    patch = generator.get_patch(str(raster), 74.8000, 13.0000, size=16)
    assert patch.shape == (16, 16)
    assert 0.0 < patch.valid_ratio < 1.0
    assert patch.padded is True
    # Padded region equals pad_value.
    assert patch.array[15, :].max() == 0.0 or patch.mask[15, :].sum() == 0


def test_out_of_bounds_raises(raster, generator):
    with pytest.raises(PatchOutOfBoundsError):
        generator.get_patch(str(raster), 75.0, 15.0, size=8)


def test_patch_size_respected(raster, generator):
    patch = generator.get_patch(str(raster), 74.801, 12.999, size=16)
    assert patch.shape == (16, 16)
    assert patch.requested_size == 16


def test_patch_metadata(raster, generator):
    patch = generator.get_patch(str(raster), 74.801, 12.999, size=8)
    assert patch.crs == "EPSG:4326"
    assert patch.resolution[0] == pytest.approx(0.0001, rel=1e-3)
    assert patch.center_lon == pytest.approx(74.801, rel=1e-6)
