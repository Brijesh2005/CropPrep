"""Tests for the R5.2.9-enriched corpus (v2.0) support in the frozen loader.

Covers:
1. ``validate_manifest`` accepts ``crop_supervised_v2.0`` while still
   rejecting unknown versions.
2. ``build_observation`` exposes the R5.2.9 DK-grid environmental features
   through ``tabular.fields`` on v2 rows and keeps the exact v1 field set for
   rows without the enrichment columns.
3. Benchmark-eligibility: the recovered (``benchmark_eligible=False``) row is
   filtered by ``FrozenCorpusLoader.validate`` so it never enters training.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from training.kaggle.frozen_corpus import (
    FrozenCorpusError,
    FrozenCorpusLoader,
    _is_benchmark_eligible,
    build_observation,
    validate_manifest,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
V2_CSV = _REPO_ROOT / "govt_crop_matched_v2" / "crop_supervised_v2.csv"
V2_MANIFEST = _REPO_ROOT / "training_manifests" / "crop_supervised_v2.0_manifest.json"
V1_MANIFEST = _REPO_ROOT / "training_manifests" / "crop_supervised_v1_manifest.json"


# --------------------------------------------------------------------------- #
# Manifest version acceptance
# --------------------------------------------------------------------------- #

def test_v2_manifest_validates() -> None:
    manifest = validate_manifest(V2_MANIFEST)
    assert manifest["dataset_version"] == "crop_supervised_v2.0"
    assert manifest["total_samples"] == 10674
    assert (manifest["train_samples"] + manifest["validation_samples"]
            + manifest["test_samples"]) == manifest["total_samples"]


def test_v1_manifest_still_validates() -> None:
    manifest = validate_manifest(V1_MANIFEST)
    assert manifest["dataset_version"] == "crop_supervised_v1.1"


def test_unknown_manifest_version_rejected(tmp_path: Path) -> None:
    data = json.loads(V1_MANIFEST.read_text(encoding="utf-8"))
    data["dataset_version"] = "crop_supervised_v9.0"
    bogus = tmp_path / "bogus_manifest.json"
    bogus.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(FrozenCorpusError):
        validate_manifest(bogus)


# --------------------------------------------------------------------------- #
# build_observation field exposure
# --------------------------------------------------------------------------- #

def _first_eligible_v2_row() -> dict:
    with open(V2_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if _is_benchmark_eligible(row):
                return row
    raise AssertionError("no benchmark-eligible row in v2 CSV")


def _mock_stam() -> MagicMock:
    stam = MagicMock()
    from training.stam.observation import SequenceInfo
    stam.resolve_sequence.return_value = SequenceInfo(pairs=[])
    return stam


def _mock_stam_windowed() -> MagicMock:
    from training.stam.config import ImageryWindowConfig
    from training.stam.observation import SequenceInfo

    stam = MagicMock()
    stam.config.imagery = ImageryWindowConfig(mode="window_days", window_days=180)
    stam.resolve_sequence_windowed.return_value = SequenceInfo(pairs=[])
    return stam


def test_windowed_imagery_path_uses_reference_date() -> None:
    """R5.3: window_days imagery mode routes build_observation through the
    windowed resolver with the row's survey date as reference."""
    row = _first_eligible_v2_row()
    stam = _mock_stam_windowed()
    build_observation(
        row, stam,
        corpus_version="crop_supervised_v2.0",
        manifest_checksum="x" * 64,
    )
    stam.resolve_sequence_windowed.assert_called_once()
    kwargs = stam.resolve_sequence_windowed.call_args.kwargs
    assert kwargs["reference_date"] is not None
    assert kwargs["reference_date"].isoformat() == (row.get("survey_date") or "")[:10]
    assert kwargs["year"] == int(row["year"])


def test_v2_row_exposes_environmental_features() -> None:
    row = _first_eligible_v2_row()
    obs = build_observation(
        row, _mock_stam(),
        corpus_version="crop_supervised_v2.0",
        manifest_checksum="x" * 64,
    )
    fields = obs.tabular.fields
    for key in [
        "lat", "lon", "spatial_match_distance_km", "season", "year",
        "kharif_ndvi", "kharif_evi", "rabi_ndvi", "rabi_evi",
        "env_match_distance_m", "soil_ph", "soil_sand_pct", "is_cropland",
        "land_cover_class", "soil_type_class",
    ]:
        assert key in fields, f"missing env field {key}"
    assert isinstance(fields["kharif_ndvi"], float)
    assert fields["season"] is not None


def test_v1_style_row_keeps_base_fields_only() -> None:
    row = {
        "record_id": "R1", "source": "government_ogd", "source_record_id": "S1",
        "source_crop_name": "A", "crop_label": "coconut", "crop_class_id": "4",
        "location_taluk": "Sullia", "location_village": "V", "location_district": "D",
        "lat": "12.5", "lon": "75.2", "year": "2020", "season": "Kharif",
        "satial_check": None,
    }
    # v1 has no enrichment columns; fields must be exactly the base five.
    row = {k: v for k, v in row.items() if k != "satial_check"}
    obs = build_observation(
        row, _mock_stam(),
        corpus_version="crop_supervised_v1.1",
        manifest_checksum="x" * 64,
    )
    assert set(obs.tabular.fields) == {
        "lat", "lon", "spatial_match_distance_km", "season", "year",
    }


# --------------------------------------------------------------------------- #
# Benchmark eligibility
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, True),
        ("True", True),
        ("true", True),
        ("1", True),
        ("yes", True),
        ("False", False),
        ("false", False),
        ("0", False),
    ],
)
def test_benchmark_eligibility(value, expected) -> None:
    row = {"benchmark_eligible": value}
    assert _is_benchmark_eligible(row) is expected


def _mini_csv(rows: list[dict]) -> Path:
    p = Path(__file__).parent / "_tmp_v2_mini.csv"
    keys = [
        "record_id", "crop_label", "crop_class_id", "location_taluk",
        "location_village", "location_district", "lat", "lon", "year",
        "season", "satellite_status", "benchmark_eligible",
    ]
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    return p


def _mini_manifest(total: int, train: int) -> Path:
    p = Path(__file__).parent / "_tmp_v2_mini_manifest.json"
    p.write_text(
        json.dumps({
            "dataset_version": "crop_supervised_v2.0",
            "total_samples": total,
            "train_samples": train,
            "validation_samples": 0,
            "test_samples": 0,
            "class_mapping": {
                "coconut": 4, "pepper": 6, "coffee": 7, "cardamom": 8, "blackgram": 9,
            },
            "class_counts": {
                "overall": {"coconut": total},
                "train": {"coconut": train},
                "validation": {},
                "test": {},
            },
            "split_strategy": "spatial_leave_one_taluk_out",
            "split_groups": {
                "train_taluk": ["Belthangady"],
                "validation_taluk": "Puttur",
                "test_taluk": "Sullia",
            },
            "supervised_classes": ["coconut"],
            "excluded_classes": [],
            "provenance_schema": {"note": "test"},
        }),
        encoding="utf-8",
    )
    return p


def test_loader_filters_non_benchmark_rows(tmp_path: Path) -> None:
    rows = [
        {"record_id": "A", "crop_label": "coconut", "crop_class_id": "4",
         "location_taluk": "Belthangady", "lat": "12.5", "lon": "75.2",
         "year": "2020", "season": "Kharif", "satellite_status": "FULL",
         "benchmark_eligible": "True"},
        {"record_id": "B", "crop_label": "coconut", "crop_class_id": "4",
         "location_taluk": "Belthangady", "lat": "12.6", "lon": "75.3",
         "year": "2020", "season": "Kharif", "satellite_status": "FULL",
         "benchmark_eligible": "True"},
        {"record_id": "C", "crop_label": "coconut", "crop_class_id": "4",
         "location_taluk": "Puttur", "lat": "12.7", "lon": "75.4",
         "year": "2020", "season": "Kharif", "satellite_status": "FULL",
         "benchmark_eligible": "False"},
    ]
    # 3 CSV rows; only 2 benchmark-eligible (both Belthangady/train).
    csv_path = _mini_csv(rows)
    manifest_path = _mini_manifest(total=2, train=2)
    loader = FrozenCorpusLoader(csv_path, manifest_path)
    loader.validate()
    assert len(loader._rows) == 2
    assert {r["record_id"] for r in loader._rows} == {"A", "B"}


def test_loader_v2_real_manifest_validates() -> None:
    loader = FrozenCorpusLoader(V2_CSV, V2_MANIFEST)
    manifest = loader.validate()
    assert len(loader._rows) == manifest["total_samples"] == 10674