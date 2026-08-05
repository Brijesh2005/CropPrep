"""Tests that shared schemas construct, dump and round-trip."""

from __future__ import annotations

from pathlib import Path

from shared.enums import CropType, FileCategory, IndexType, Resolution, Severity
from shared.schemas import (
    ConfigMetadataSchema,
    DatasetInventorySchema,
    DatasetSummarySchema,
    FileEntrySchema,
    ImageDatasetLocationSchema,
    ImageDatasetRecordSchema,
    MetadataRecordSchema,
    PredictionInputSchema,
    PredictionResultSchema,
    RasterMetadataSchema,
    ReleaseMetadataSchema,
    TrainingRunSchema,
    ValidationIssueSchema,
    ValidationReportSchema,
)


def test_dataset_inventory_counts(tmp_path) -> None:
    inv = DatasetInventorySchema(
        root=tmp_path,
        entries=[
            FileEntrySchema(
                path=tmp_path / "a.tif",
                relative_path="a.tif",
                category=FileCategory.GEOTIFF,
                index_type=IndexType.NDVI,
                resolution=Resolution.R10M,
                year=2020,
            ),
            FileEntrySchema(
                path=tmp_path / "b.tif",
                relative_path="b.tif",
                category=FileCategory.GEOTIFF,
                index_type=IndexType.EVI,
                resolution=Resolution.R20M,
                year=2021,
            ),
            FileEntrySchema(
                path=tmp_path / "c.csv",
                relative_path="c.csv",
                category=FileCategory.CSV,
            ),
        ],
    )
    counts = inv.counts()
    assert counts["total"] == 3
    assert counts["csv"] == 1
    assert counts["geotiff"] == 2
    assert counts["ndvi"] == 1
    assert counts["evi"] == 1
    assert counts["r10m"] == 1
    assert counts["r20m"] == 1


def test_dataset_summary_roundtrip(tmp_path) -> None:
    summary = DatasetSummarySchema(
        name="crop-ds",
        root=tmp_path,
        total_files=10,
        years_covered=[2020, 2021],
    )
    data = summary.to_dict()
    assert data["name"] == "crop-ds"
    assert data["years_covered"] == [2020, 2021]
    assert isinstance(data["root"], str)


def test_raster_metadata(tmp_path) -> None:
    raster = RasterMetadataSchema(
        path=tmp_path / "r.tif",
        filename="r.tif",
        width=100,
        height=50,
        bands=1,
        crs="EPSG:32643",
        pixel_size=(10.0, 10.0),
    )
    data = raster.to_dict()
    assert data["pixel_size"] == [10.0, 10.0]
    assert data["crs"] == "EPSG:32643"


def test_image_catalog_records() -> None:
    loc = ImageDatasetLocationSchema(
        path=Path("raw/2020/NDVI.tif"),
        index_type=IndexType.NDVI,
        resolution=Resolution.R10M,
        year=2020,
    )
    assert loc.to_dict()["index_type"] == "NDVI"
    rec = ImageDatasetRecordSchema(handle="h-1", path=Path("raw/2020/NDVI.tif"))
    assert rec.to_dict()["handle"] == "h-1"


def test_prediction_roundtrip() -> None:
    inp = PredictionInputSchema(location="somewhere", crop=CropType.MAIZE, year=2020)
    data = inp.to_dict()
    assert data["crop"] == "maize"

    res = PredictionResultSchema(location="somewhere", predicted_yield=8.4, confidence=0.9)
    assert res.to_dict()["predicted_yield"] == 8.4


def test_validation_report_by_severity() -> None:
    report = ValidationReportSchema(
        root="raw",
        passed=False,
        issues=[
            ValidationIssueSchema(severity=Severity.ERROR, code="X", category="csv", message="bad"),
            ValidationIssueSchema(severity=Severity.WARNING, code="Y", category="csv", message="warn"),
        ],
    )
    assert report.by_severity() == {"error": 1, "warning": 1}
    assert report.to_dict()["passed"] is False


def test_metadata_record_to_dict(tmp_path) -> None:
    rec = MetadataRecordSchema(
        path=tmp_path / "x.csv",
        relative_path="x.csv",
        category=FileCategory.CSV,
        row_count=42,
    )
    data = rec.to_dict()
    assert data["row_count"] == 42
    assert data["category"] == "csv"
    assert "path" in data
    assert "path" not in rec.to_dict(include_path=False)


def test_config_metadata() -> None:
    config = ConfigMetadataSchema(source="settings.yaml", env_prefix="DM_")
    assert config.to_dict()["source"] == "settings.yaml"


def test_training_run() -> None:
    run = TrainingRunSchema(run_id="r-1", model_name="model-x", epoch=3, metrics={"acc": 0.9})
    assert run.to_dict()["metrics"]["acc"] == 0.9


def test_release_metadata() -> None:
    release = ReleaseMetadataSchema(kind="application", name="app", version="1.0.0")
    assert release.to_dict()["version"] == "1.0.0"
    assert release.kind == "application"
