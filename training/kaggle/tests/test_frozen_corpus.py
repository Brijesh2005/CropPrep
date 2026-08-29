"""Frozen corpus adapter tests — R5.2.7 → R5.2.8 integration.

Tests cover:
- Manifest validation (schema, version, counts, class mapping)
- CSV loading and schema validation
- Observation construction (location, temporal, crop, provenance)
- Split assignment (taluk-based R5.2.7 spatial split)
- Data contract printout and verification
- Guardrails: no old data leakage, no non-frozen observations
"""

from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from training.kaggle.frozen_corpus import (
    FrozenCorpusError,
    FrozenCorpusLoader,
    _TALUK_SPLIT,
    _build_location,
    _build_temporal,
    _determine_split,
    _load_csv,
    build_observation,
    validate_manifest,
)
from training.stam.observation import (
    ImagePairRef,
    ImageRecordRef,
    QualityReport,
    SequenceInfo,
)


def _pair_sequence() -> SequenceInfo:
    """A minimal valid sequence with one NDVI/EVI pair."""
    return SequenceInfo(pairs=[ImagePairRef(date=date(2020, 7, 1))])


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture()
def manifest_path(tmp_path: Path) -> Path:
    """Write a minimal valid manifest to disk."""
    manifest = {
        "dataset_version": "crop_supervised_v1.1",
        "creation_timestamp": "2026-08-17T16:02:06.653645",
        "source_datasets": {
            "government_ogd": {
                "name": "Karnataka Crop Survey OGD",
                "total_raw_records": 199345,
                "valid_matched_records": 10,
            }
        },
        "total_samples": 10,
        "train_samples": 6,
        "validation_samples": 2,
        "test_samples": 2,
        "class_mapping": {
            "coconut": 4,
            "pepper": 6,
            "coffee": 7,
            "cardamom": 8,
            "blackgram": 9,
        },
        "class_counts": {
            "overall": {
                "coconut": 5,
                "pepper": 3,
                "coffee": 1,
                "cardamom": 1,
            },
            "train": {"coconut": 3, "pepper": 2, "coffee": 1},
            "validation": {"coconut": 1, "pepper": 1},
            "test": {"coconut": 1, "cardamom": 1},
        },
        "class_weights": {},
        "split_strategy": "spatial_leave_one_taluk_out",
        "split_groups": {
            "train_taluk": ["Belthangady", "Mangalore", "Puttur"],
            "validation_taluk": "Bantwal",
            "test_taluk": "Sullia",
        },
        "excluded_classes": [],
        "evaluation_policy": {},
        "feature_schema": {
            "tabular": ["NDVI", "EVI"],
            "satellite": "sentinel2",
            "temporal": ["year", "season"],
            "spatial": ["lat", "lon", "village", "taluk", "district"],
        },
        "provenance_schema": {
            "government_ogd": "Full pipeline",
        },
        "reproducibility": {
            "random_seed": 42,
            "code_version": "R5.2.7",
            "dataset_checksums": {
                "crop_supervised_v1.csv": "test_checksum",
            },
        },
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


@pytest.fixture()
def csv_path(tmp_path: Path) -> Path:
    """Write a minimal valid CSV to disk (10 rows matching the manifest)."""
    rows = [
        {
            "record_id": f"gov_TALUK{i}_VILLAGE{i}_2020_Kharif_crop_{i}",
            "source": "government_ogd",
            "source_record_id": f"SRC_{i}",
            "crop_label": label,
            "crop_class_id": class_id,
            "source_crop_name": label.title(),
            "location_hobli": f"H{i}",
            "location_taluk": taluk,
            "location_village": f"VILLAGE{i}",
            "location_district": "Dakshina Kannada",
            "lat": f"12.{i:06d}",
            "lon": f"75.{i:06d}",
            "year": 2020 if i < 5 else 2021,
            "season": "Kharif" if i % 2 == 0 else "Rabi",
            "survey_date": f"2020-09-{i+1:02d}",
            "spatial_match_distance_km": "0.1",
            "temporal_match_status": "EXACT_SEASON",
            "tabular_source": "district_grid",
            "image_source": "sentinel2",
            "ndvi_available": "True",
            "evi_available": "True",
            "satellite_status": "FULL",
        }
        for i, (label, class_id, taluk) in enumerate(
            [
                ("coconut", 4, "Belthangady"),   # train
                ("coconut", 4, "Mangalore"),      # train
                ("coconut", 4, "Puttur"),         # train
                ("pepper", 6, "Belthangady"),     # train
                ("pepper", 6, "Mangalore"),       # train
                ("coffee", 7, "Puttur"),          # train
                ("coconut", 4, "Bantwal"),        # val
                ("pepper", 6, "Bantwal"),         # val
                ("coconut", 4, "Sullia"),         # test
                ("cardamom", 8, "Sullia"),        # test
            ]
        )
    ]
    path = tmp_path / "crop_supervised_v1.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.fixture()
def mock_stam() -> MagicMock:
    """A mock STAM instance that returns single-pair sequences."""
    stam = MagicMock()
    stam.resolve_sequence.return_value = _pair_sequence()
    return stam


# --------------------------------------------------------------------------- #
# Manifest validation
# --------------------------------------------------------------------------- #


class TestManifestValidation:
    def test_valid_manifest(self, manifest_path: Path) -> None:
        result = validate_manifest(manifest_path)
        assert result["dataset_version"] == "crop_supervised_v1.1"
        assert result["total_samples"] == 10
        assert result["train_samples"] == 6

    def test_missing_manifest(self, tmp_path: Path) -> None:
        with pytest.raises(FrozenCorpusError, match="not found"):
            validate_manifest(tmp_path / "nonexistent.json")

    def test_wrong_version(self, manifest_path: Path, tmp_path: Path) -> None:
        manifest = json.loads(manifest_path.read_text())
        manifest["dataset_version"] = "old_version"
        bad = tmp_path / "bad_manifest.json"
        bad.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(FrozenCorpusError, match="Unexpected manifest version"):
            validate_manifest(bad)

    def test_split_counts_mismatch(self, manifest_path: Path, tmp_path: Path) -> None:
        manifest = json.loads(manifest_path.read_text())
        manifest["train_samples"] = 999  # does not sum to total
        bad = tmp_path / "bad_manifest.json"
        bad.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(FrozenCorpusError, match="do not sum to total_samples"):
            validate_manifest(bad)

    def test_class_mapping_mismatch(self, manifest_path: Path, tmp_path: Path) -> None:
        manifest = json.loads(manifest_path.read_text())
        manifest["class_mapping"]["new_crop"] = 99
        bad = tmp_path / "bad_manifest.json"
        bad.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(FrozenCorpusError, match="Class mapping mismatch"):
            validate_manifest(bad)

    def test_missing_split_groups(self, manifest_path: Path, tmp_path: Path) -> None:
        manifest = json.loads(manifest_path.read_text())
        del manifest["split_groups"]["train_taluk"]
        bad = tmp_path / "bad_manifest.json"
        bad.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(FrozenCorpusError, match="train_taluk is missing"):
            validate_manifest(bad)


# --------------------------------------------------------------------------- #
# CSV loading
# --------------------------------------------------------------------------- #


class TestCSVLoading:
    def test_load_valid_csv(self, csv_path: Path) -> None:
        rows = _load_csv(csv_path)
        assert len(rows) == 10
        assert rows[0]["crop_label"] == "coconut"
        assert rows[0]["location_taluk"] == "Belthangady"

    def test_missing_csv(self, tmp_path: Path) -> None:
        with pytest.raises(FrozenCorpusError, match="not found"):
            _load_csv(tmp_path / "nonexistent.csv")

    def test_empty_csv(self, tmp_path: Path) -> None:
        header = (
            "record_id,source,source_record_id,crop_label,crop_class_id,"
            "source_crop_name,location_hobli,location_taluk,location_village,"
            "location_district,lat,lon,year,season,survey_date,"
            "spatial_match_distance_km,temporal_match_status,tabular_source,"
            "image_source,ndvi_available,evi_available,satellite_status"
        )
        path = tmp_path / "empty.csv"
        path.write_text(header + "\n", encoding="utf-8")
        with pytest.raises(FrozenCorpusError, match="zero data rows"):
            _load_csv(path)

    def test_missing_columns(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.csv"
        path.write_text("record_id,foo\nrow1,bar\n", encoding="utf-8")
        with pytest.raises(FrozenCorpusError, match="missing required columns"):
            _load_csv(path)


# --------------------------------------------------------------------------- #
# Split assignment
# --------------------------------------------------------------------------- #


class TestSplitAssignment:
    def test_train_taluks(self) -> None:
        assert _determine_split({"location_taluk": "Belthangady"}) == "train"
        assert _determine_split({"location_taluk": "Mangalore"}) == "train"
        assert _determine_split({"location_taluk": "Puttur"}) == "train"

    def test_val_taluk(self) -> None:
        assert _determine_split({"location_taluk": "Bantwal"}) == "val"

    def test_test_taluk(self) -> None:
        assert _determine_split({"location_taluk": "Sullia"}) == "test"

    def test_unknown_taluk(self) -> None:
        assert _determine_split({"location_taluk": "Unknown"}) == "unknown"

    def test_missing_taluk(self) -> None:
        assert _determine_split({}) == "unknown"


# --------------------------------------------------------------------------- #
# Observation construction
# --------------------------------------------------------------------------- #


class TestObservationConstruction:
    def test_build_location(self) -> None:
        row = {
            "lon": "75.123",
            "lat": "12.456",
            "record_id": "rec_001",
            "location_hobli": "Hobli",
            "location_village": "Village",
            "location_taluk": "Taluk",
            "location_district": "District",
            "spatial_match_distance_km": "0.5",
        }
        loc = _build_location(row)
        assert loc.lon == 75.123
        assert loc.lat == 12.456
        assert loc.admin.village == "Village"
        assert loc.admin.taluk == "Taluk"
        assert loc.admin.district == "District"
        assert loc.distance_km == 0.5

    def test_build_temporal(self) -> None:
        row = {"year": "2021", "season": "Rabi", "survey_date": "2021-01-15"}
        temp = _build_temporal(row)
        assert temp.year == 2021
        assert temp.season == "Rabi"
        assert len(temp.observation_dates) == 1

    def test_build_observation_provenance(self, mock_stam: MagicMock) -> None:
        row = {
            "record_id": "gov_TEST_001",
            "source": "government_ogd",
            "source_record_id": "SRC_001",
            "crop_label": "coconut",
            "crop_class_id": "4",
            "source_crop_name": "Coconut",
            "location_hobli": "H",
            "location_taluk": "Belthangady",
            "location_village": "V",
            "location_district": "Dakshina Kannada",
            "lat": "12.0",
            "lon": "75.0",
            "year": "2020",
            "season": "Kharif",
            "survey_date": "2020-09-01",
            "spatial_match_distance_km": "0.1",
            "temporal_match_status": "EXACT_SEASON",
            "tabular_source": "district_grid",
            "image_source": "sentinel2",
            "ndvi_available": "True",
            "evi_available": "True",
            "satellite_status": "FULL",
        }
        obs = build_observation(
            row, mock_stam,
            corpus_version="crop_supervised_v1.1",
            manifest_checksum="abc123",
        )
        assert obs.crop == "coconut"
        assert obs.quality.passed is True
        assert obs.quality.overall_score == 100.0
        assert obs.provenance["corpus"] == "crop_supervised_v1"
        assert obs.provenance["record_id"] == "gov_TEST_001"
        assert obs.provenance["split"] == "train"
        assert obs.provenance["satellite_status"] == "FULL"
        assert obs.tabular.matched_level == "frozen_corpus"

    def test_no_yield_in_frozen_corpus(self, mock_stam: MagicMock) -> None:
        row = {
            "record_id": "gov_TEST_002",
            "source": "government_ogd",
            "source_record_id": "SRC_002",
            "crop_label": "pepper",
            "crop_class_id": "6",
            "source_crop_name": "Pepper",
            "location_hobli": "H",
            "location_taluk": "Bantwal",
            "location_village": "V",
            "location_district": "Dakshina Kannada",
            "lat": "12.0",
            "lon": "75.0",
            "year": "2020",
            "season": "Kharif",
            "survey_date": "2020-09-01",
            "spatial_match_distance_km": "0.1",
            "temporal_match_status": "EXACT_SEASON",
            "tabular_source": "district_grid",
            "image_source": "sentinel2",
            "ndvi_available": "True",
            "evi_available": "True",
            "satellite_status": "FULL",
        }
        obs = build_observation(
            row, mock_stam,
            corpus_version="crop_supervised_v1.1",
            manifest_checksum="abc123",
        )
        assert obs.yield_value is None


# --------------------------------------------------------------------------- #
# FrozenCorpusLoader — end-to-end
# --------------------------------------------------------------------------- #


class TestFrozenCorpusLoader:
    def test_validate(self, csv_path: Path, manifest_path: Path) -> None:
        loader = FrozenCorpusLoader(csv_path, manifest_path)
        manifest = loader.validate()
        assert manifest["total_samples"] == 10

    def test_build_returns_split_lists(
        self, csv_path: Path, manifest_path: Path, mock_stam: MagicMock
    ) -> None:
        loader = FrozenCorpusLoader(csv_path, manifest_path)
        train, val, test = loader.build(mock_stam)
        assert len(train) == 6
        assert len(val) == 2
        assert len(test) == 2

    def test_build_all_observations_are_frozen(
        self, csv_path: Path, manifest_path: Path, mock_stam: MagicMock
    ) -> None:
        loader = FrozenCorpusLoader(csv_path, manifest_path)
        train, val, test = loader.build(mock_stam)
        for obs in train + val + test:
            assert obs.provenance.get("corpus") == "crop_supervised_v1"

    def test_build_records_build_stats(
        self, csv_path: Path, manifest_path: Path, mock_stam: MagicMock
    ) -> None:
        loader = FrozenCorpusLoader(csv_path, manifest_path)
        loader.build(mock_stam)
        assert loader.last_build_stats == {
            "rows": 10,
            "excluded": 0,
            "train": 6,
            "val": 2,
            "test": 2,
            "accepted": 10,
        }

    def test_build_splits_all_non_empty(
        self, csv_path: Path, manifest_path: Path, mock_stam: MagicMock
    ) -> None:
        """Regression: R5.2 requires every split to be non-empty. An empty
        val split (the 8601/0/1518 contradiction) must never reach Phase 4 —
        the hold-out guard in Experiment now fails loudly."""
        loader = FrozenCorpusLoader(csv_path, manifest_path)
        train, val, test = loader.build(mock_stam)
        assert train and val and test
        assert loader.last_build_stats["accepted"] == (
            loader.last_build_stats["train"]
            + loader.last_build_stats["val"]
            + loader.last_build_stats["test"]
        )

    def test_imagery_summary_structure(
        self, csv_path: Path, manifest_path: Path, mock_stam: MagicMock
    ) -> None:
        loader = FrozenCorpusLoader(csv_path, manifest_path)
        train, val, test = loader.build(mock_stam)
        summary = loader.imagery_summary(train, val, test, max_observations=8)
        assert summary["train"] == 6
        assert summary["val"] == 2
        assert summary["test"] == 2
        assert summary["accepted"] == 10
        # mock sequences carry one pair each but no NDVI/EVI refs -> partial.
        assert summary["partial_pairs"] == 10
        assert summary["fully_paired"] == 0
        assert summary["observations_min"] == 1
        assert summary["observations_max"] == 1
        assert summary["at_max_cap"] == 0
        assert summary["rows"] == 10
        assert summary["excluded"] == 0
        assert summary["patch_sizes"] == []

    def test_corpus_imagery_diagnostics_slot_counts(
        self, csv_path: Path, manifest_path: Path
    ) -> None:
        """R5.3: slot-level real-vs-zero-filled statistics match the sequence
        lengths: 10 samples x 8 slots, one real NDVI/EVI ref pair each ->
        10 real slots and 70 zero-filled window slots per stream, never a
        sample trained entirely on zero-fill."""
        loader = FrozenCorpusLoader(csv_path, manifest_path)
        stam = MagicMock()
        stam.resolve_sequence.return_value = _real_pair_sequence(1)
        train, val, test = loader.build(stam)
        diag = loader.corpus_imagery_diagnostics(
            train, val, test, max_observations=8
        )
        assert diag["max_observations"] == 8
        for part in ("train", "val", "test"):
            assert part in diag
        overall = diag["overall"]
        assert overall["samples"] == 10
        for stream in ("ndvi", "evi"):
            s = overall["streams"][stream]
            assert s["total_slots"] == 80
            assert s["real_slots"] == 10
            assert s["zero_filled_slots"] == 70
            assert s["real_frac"] == pytest.approx(0.125)
            assert s["samples_with_real_imagery"] == 10
            assert s["samples_without_imagery"] == 0

    def test_corpus_imagery_diagnostics_flags_all_zero_samples(self) -> None:
        """A sample with an empty imagery sequence surfaces as
        ``samples_without_imagery`` instead of being silently zero-filled."""
        loader = object.__new__(FrozenCorpusLoader)
        obs = MagicMock()
        obs.sequence = SequenceInfo(pairs=[])
        diag = loader.corpus_imagery_diagnostics([obs], [], [], max_observations=4)
        s = diag["overall"]["streams"]["ndvi"]
        assert s["total_slots"] == 4
        assert s["real_slots"] == 0
        assert s["samples_without_imagery"] == 1
        assert diag["overall"]["streams"]["evi"]["samples_without_imagery"] == 1

    def test_build_excludes_empty_sequences(
        self, csv_path: Path, manifest_path: Path, mock_stam: MagicMock
    ) -> None:
        """Rows resolving to zero NDVI/EVI pairs must be dropped at build
        time — otherwise the preprocessing quality gate rejects them
        mid-epoch and crashes the DataLoader."""
        loader = FrozenCorpusLoader(csv_path, manifest_path)
        calls = {"n": 0}

        def resolve(lon: float, lat: float, **kwargs: Any) -> SequenceInfo:
            calls["n"] += 1
            if calls["n"] == 1:
                return SequenceInfo(pairs=[])
            return _pair_sequence()

        mock_stam.resolve_sequence.side_effect = resolve

        train, val, test = loader.build(mock_stam)
        all_obs = train + val + test
        assert len(all_obs) == 9
        assert all(o.num_observations() >= 1 for o in all_obs)

    def test_data_contract_printout(
        self, csv_path: Path, manifest_path: Path, mock_stam: MagicMock
    ) -> None:
        loader = FrozenCorpusLoader(csv_path, manifest_path)
        train, val, test = loader.build(mock_stam)
        contract = loader.data_contract_printout(train, val, test)
        assert contract["total_samples"] == 10
        assert contract["train_samples"] == 6
        assert contract["val_samples"] == 2
        assert contract["test_samples"] == 2
        assert "coconut" in contract["overall_class_counts"]

    def test_verify_contract_passes(
        self, csv_path: Path, manifest_path: Path, mock_stam: MagicMock
    ) -> None:
        loader = FrozenCorpusLoader(csv_path, manifest_path)
        train, val, test = loader.build(mock_stam)
        contract = loader.data_contract_printout(train, val, test)
        passed, errors = loader.verify_contract(contract, train, val, test)
        assert passed is True
        assert errors == []

    def test_verify_contract_detects_count_mismatch(
        self, csv_path: Path, manifest_path: Path, mock_stam: MagicMock
    ) -> None:
        loader = FrozenCorpusLoader(csv_path, manifest_path)
        train, val, test = loader.build(mock_stam)
        contract = loader.data_contract_printout(train, val, test)
        # Simulate a count mismatch by removing an observation.
        passed, errors = loader.verify_contract(contract, train[:-1], val, test)
        assert passed is False
        assert any("Train count mismatch" in e for e in errors)


# --------------------------------------------------------------------------- #
# Guardrails: no old data leakage
# --------------------------------------------------------------------------- #


class TestGuardrails:
    def test_no_non_frozen_provenance(
        self, csv_path: Path, manifest_path: Path, mock_stam: MagicMock
    ) -> None:
        """All observations must carry frozen corpus provenance."""
        loader = FrozenCorpusLoader(csv_path, manifest_path)
        train, val, test = loader.build(mock_stam)
        all_obs = train + val + test
        for obs in all_obs:
            assert "corpus" in obs.provenance
            assert obs.provenance["corpus"] == "crop_supervised_v1"

    def test_no_old_data_source(
        self, csv_path: Path, manifest_path: Path, mock_stam: MagicMock
    ) -> None:
        """No observation should come from data_season.csv or other old sources."""
        loader = FrozenCorpusLoader(csv_path, manifest_path)
        train, val, test = loader.build(mock_stam)
        for obs in train + val + test:
            assert "source" in obs.provenance
            assert obs.provenance["source"] == "government_ogd"

    def test_manifest_checksum_is_stamped(
        self, csv_path: Path, manifest_path: Path, mock_stam: MagicMock
    ) -> None:
        """Every observation must carry the manifest SHA-256 checksum."""
        loader = FrozenCorpusLoader(csv_path, manifest_path)
        train, val, test = loader.build(mock_stam)
        for obs in train + val + test:
            assert "manifest_checksum" in obs.provenance
            assert len(obs.provenance["manifest_checksum"]) == 64  # SHA-256 hex


# --------------------------------------------------------------------------- #
# Experiment pre_split integration
# --------------------------------------------------------------------------- #


class TestExperimentPreSplit:
    def test_experiment_accepts_pre_split(self) -> None:
        """Experiment stores pre_split and uses it in _holdout_split."""
        from training.training.config import TrainingConfig
        from training.training.experiment import Experiment

        config = TrainingConfig()
        obs = [MagicMock()]  # dummy
        train_obs = [MagicMock(), MagicMock()]
        val_obs = [MagicMock()]
        test_obs = [MagicMock()]

        experiment = Experiment(
            config, obs,
            pre_split=(train_obs, val_obs, test_obs),
        )
        assert experiment._pre_split is not None
        t, v, te = experiment._holdout_split()
        assert t == train_obs
        assert v == val_obs
        assert te == test_obs

    def test_experiment_without_pre_split_uses_splitter(self) -> None:
        """Without pre_split, Experiment falls back to split_observations."""
        from training.training.config import TrainingConfig
        from training.training.experiment import Experiment

        config = TrainingConfig()
        obs = [MagicMock()] * 10
        experiment = Experiment(config, obs)
        assert experiment._pre_split is None


# --------------------------------------------------------------------------- #
# Multimodal contract
# --------------------------------------------------------------------------- #


class TestMultimodalContract:
    def test_observation_has_all_modalities(
        self, csv_path: Path, manifest_path: Path, mock_stam: MagicMock
    ) -> None:
        """Every observation must have tabular, temporal, sequence, quality."""
        loader = FrozenCorpusLoader(csv_path, manifest_path)
        train, val, test = loader.build(mock_stam)
        for obs in train + val + test:
            assert obs.location is not None, "missing location"
            assert obs.temporal is not None, "missing temporal"
            assert obs.tabular is not None, "missing tabular"
            assert obs.sequence is not None, "missing sequence"
            assert obs.quality is not None, "missing quality"

    def test_tabular_features_present(self, mock_stam: MagicMock) -> None:
        """Tabular features dict is present on every observation."""
        row = _minimal_row("Belthangady", "coconut", 4)
        obs = build_observation(
            row, mock_stam,
            corpus_version="crop_supervised_v1.1",
            manifest_checksum="abc123",
        )
        assert isinstance(obs.tabular.fields, dict)
        assert obs.tabular.matched_level == "frozen_corpus"

    def test_temporal_info_populated(self, mock_stam: MagicMock) -> None:
        """Temporal info has year and season."""
        row = _minimal_row("Bantwal", "pepper", 6)
        row["year"] = "2021"
        row["season"] = "Rabi"
        obs = build_observation(
            row, mock_stam,
            corpus_version="crop_supervised_v1.1",
            manifest_checksum="abc123",
        )
        assert obs.temporal.year == 2021
        assert obs.temporal.season == "Rabi"

    def test_quality_report_passed(self, mock_stam: MagicMock) -> None:
        """Quality report is pre-validated (passed=True, score=100)."""
        row = _minimal_row("Sullia", "coffee", 7)
        obs = build_observation(
            row, mock_stam,
            corpus_version="crop_supervised_v1.1",
            manifest_checksum="abc123",
        )
        assert obs.quality.passed is True
        assert obs.quality.overall_score == 100.0

    def test_location_resolved(self, mock_stam: MagicMock) -> None:
        """LocationInfo has lon, lat, admin hierarchy."""
        row = _minimal_row("Puttur", "cardamom", 8)
        row["lat"] = "12.764"
        row["lon"] = "75.983"
        obs = build_observation(
            row, mock_stam,
            corpus_version="crop_supervised_v1.1",
            manifest_checksum="abc123",
        )
        assert obs.location.lon == 75.983
        assert obs.location.lat == 12.764
        assert obs.location.admin is not None
        assert obs.location.admin.taluk == "Puttur"


# --------------------------------------------------------------------------- #
# Duplicate identity
# --------------------------------------------------------------------------- #


class TestDuplicateIdentity:
    def test_no_duplicate_record_ids_in_csv(self, csv_path: Path) -> None:
        rows = _load_csv(csv_path)
        ids = [r["record_id"] for r in rows]
        assert len(ids) == len(set(ids)), "duplicate record_ids found"

    def test_no_duplicate_record_ids_in_observations(
        self, csv_path: Path, manifest_path: Path, mock_stam: MagicMock
    ) -> None:
        loader = FrozenCorpusLoader(csv_path, manifest_path)
        train, val, test = loader.build(mock_stam)
        all_obs = train + val + test
        provenance_ids = [obs.provenance.get("record_id") for obs in all_obs]
        assert len(provenance_ids) == len(set(provenance_ids)), (
            "duplicate record_ids in observations"
        )


# --------------------------------------------------------------------------- #
# Split class distribution
# --------------------------------------------------------------------------- #


class TestSplitClassDistribution:
    def test_train_class_counts(self, csv_path: Path, manifest_path: Path) -> None:
        from training.kaggle.frozen_corpus import _determine_split

        rows = _load_csv(csv_path)
        train_classes: dict[str, int] = {}
        for row in rows:
            if _determine_split(row) == "train":
                label = row.get("crop_label", "unknown")
                train_classes[label] = train_classes.get(label, 0) + 1
        assert train_classes.get("coconut", 0) > 0
        assert train_classes.get("pepper", 0) > 0

    def test_val_has_only_bantwal(self, csv_path: Path) -> None:
        from training.kaggle.frozen_corpus import _determine_split

        rows = _load_csv(csv_path)
        for row in rows:
            if _determine_split(row) == "val":
                assert row["location_taluk"] == "Bantwal"

    def test_test_has_only_sullia(self, csv_path: Path) -> None:
        from training.kaggle.frozen_corpus import _determine_split

        rows = _load_csv(csv_path)
        for row in rows:
            if _determine_split(row) == "test":
                assert row["location_taluk"] == "Sullia"


# --------------------------------------------------------------------------- #
# Observation construction completeness
# --------------------------------------------------------------------------- #


class TestObservationConstructionCompleteness:
    def test_all_required_fields_present(self, mock_stam: MagicMock) -> None:
        """All required AgriculturalObservation fields are populated."""
        row = _minimal_row("Belthangady", "coconut", 4)
        obs = build_observation(
            row, mock_stam,
            corpus_version="crop_supervised_v1.1",
            manifest_checksum="abc123",
        )
        assert obs.observation_id is not None
        assert obs.created_at is not None
        assert obs.location is not None
        assert obs.temporal is not None
        assert obs.tabular is not None
        assert obs.sequence is not None
        assert obs.quality is not None
        assert obs.crop == "coconut"
        assert obs.yield_value is None
        assert obs.provenance != {}
        assert obs.dataset_version == "crop_supervised_v1.1"

    def test_crop_class_id_in_provenance(self, mock_stam: MagicMock) -> None:
        row = _minimal_row("Mangalore", "pepper", 6)
        obs = build_observation(
            row, mock_stam,
            corpus_version="crop_supervised_v1.1",
            manifest_checksum="abc123",
        )
        assert obs.provenance["crop_class_id"] == 6

    def test_satellite_status_in_provenance(self, mock_stam: MagicMock) -> None:
        row = _minimal_row("Sullia", "coffee", 7)
        obs = build_observation(
            row, mock_stam,
            corpus_version="crop_supervised_v1.1",
            manifest_checksum="abc123",
        )
        assert obs.provenance["satellite_status"] == "FULL"

    def test_ndvi_evi_flags_in_provenance(self, mock_stam: MagicMock) -> None:
        row = _minimal_row("Bantwal", "coconut", 4)
        obs = build_observation(
            row, mock_stam,
            corpus_version="crop_supervised_v1.1",
            manifest_checksum="abc123",
        )
        assert obs.provenance["ndvi_available"] == "True"
        assert obs.provenance["evi_available"] == "True"


# --------------------------------------------------------------------------- #
# Helper: build a minimal valid CSV row
# --------------------------------------------------------------------------- #


def _minimal_row(taluk: str, crop_label: str, crop_class_id: int) -> dict[str, str]:
    """Return a minimal valid frozen-corpus CSV row dict."""
    return {
        "record_id": f"gov_TEST_{taluk}_{crop_label}_001",
        "source": "government_ogd",
        "source_record_id": "SRC_TEST_001",
        "crop_label": crop_label,
        "crop_class_id": str(crop_class_id),
        "source_crop_name": crop_label.title(),
        "location_hobli": "Hobli",
        "location_taluk": taluk,
        "location_village": "Village",
        "location_district": "Dakshina Kannada",
        "lat": "12.500",
        "lon": "75.500",
        "year": "2020",
        "season": "Kharif",
        "survey_date": "2020-09-01",
        "spatial_match_distance_km": "0.1",
        "temporal_match_status": "EXACT_SEASON",
        "tabular_source": "district_grid",
        "image_source": "sentinel2",
        "ndvi_available": "True",
        "evi_available": "True",
        "satellite_status": "FULL",
    }


# --------------------------------------------------------------------------- #
# Root-cause regression tests (ERROR 1 / ERROR 3)
# --------------------------------------------------------------------------- #


def _real_pair_sequence(n_dates: int = 2) -> SequenceInfo:
    """A sequence with real NDVI/EVI record refs so patches can be read."""
    import numpy as np

    pairs = []
    for i in range(n_dates):
        day = 1 + i
        pairs.append(
            ImagePairRef(
                date=date(2020, 6, day),
                ndvi=ImageRecordRef(
                    path=f"/img/ndvi_{day}.tif",
                    relative_path=f"ndvi_{day}.tif",
                    index_type="NDVI",
                    resolution="R10m",
                ),
                evi=ImageRecordRef(
                    path=f"/img/evi_{day}.tif",
                    relative_path=f"evi_{day}.tif",
                    index_type="EVI",
                    resolution="R10m",
                ),
            )
        )
    del np
    return SequenceInfo(pairs=pairs, resolution="R10m", crs="EPSG:4326")


class TestFrozenPreprocessorShapes:
    """ERROR 1: image patches + zero-fill must share the config (224) size.

    The frozen corpus previously hard-coded ``patch_size=128`` while the
    preprocessing config uses 224, so real patches (resized to 224) and the
    zero-fill fallback (128) collided in ``collate_samples`` -> torch.stack.
    """

    def _fit_obs(self, rows: dict[str, str] | list[dict[str, str]], pairs: SequenceInfo) -> Any:
        import numpy as np
        from training.preprocessing.master_pipeline import Preprocessor

        row_list = rows if isinstance(rows, list) else [rows]
        stam = MagicMock()
        stam.resolve_sequence.return_value = pairs
        objs = [
            build_observation(
                r,
                stam,
                corpus_version="crop_supervised_v1.1",
                manifest_checksum="abc123",
            )
            for r in row_list
        ]

        def extractor(path, lon, lat, size=224):
            from training.stam.patch_generator import RasterPatch

            return RasterPatch(
                path=path,
                array=np.full((size, size), 0.42, dtype="float32"),
                mask=np.ones((size, size), dtype=bool),
                requested_size=size,
                window=(0, 0, size, size),
                bounds=(lon - 1, lat - 1, lon + 1, lat + 1),
                crs="EPSG:4326",
            )

        cfg = Preprocessor.from_config(
            Path(__file__).resolve().parents[3] / "training" / "config" / "preprocessing.yaml"
        )
        pre = cfg.fit(objs, extractor=extractor)
        return pre, objs

    def test_real_patch_shape_is_config_size(self) -> None:
        """Real patches -> [T,1,224,224], never [T,1,128,128]."""
        import numpy as np
        from training.stam.patch_generator import RasterPatch

        row = _minimal_row("Belthangady", "coconut", 4)
        pre, objs = self._fit_obs(row, _real_pair_sequence())

        def extractor(path, lon, lat, size=224):
            return RasterPatch(
                path=path,
                array=np.full((size, size), 0.42, dtype="float32"),
                mask=np.ones((size, size), dtype=bool),
                requested_size=size,
                window=(0, 0, size, size),
                bounds=(lon - 1, lat - 1, lon + 1, lat + 1),
                crs="EPSG:4326",
            )

        sample = pre.transform(objs[0], extractor=extractor)
        assert sample["ndvi"].shape[0] == pre.config.temporal.max_observations
        assert sample["ndvi"].shape[1:] == (1, pre.config.image.size, pre.config.image.size)
        assert sample["evi"].shape == sample["ndvi"].shape

    def test_zero_fill_fallback_shape_matches_config(self) -> None:
        """When all patches fail, zero-fill must be [T,1,224,224], not 128."""
        from training.preprocessing.master_pipeline import Preprocessor

        row = _minimal_row("Mangalore", "pepper", 6)
        stam = MagicMock()
        stam.resolve_sequence.return_value = _real_pair_sequence()

        def boom_extractor(path, lon, lat, size=224):
            raise ValueError("no imagery mounted")

        pre = Preprocessor.from_config(
            Path(__file__).resolve().parents[3] / "training" / "config" / "preprocessing.yaml"
        )
        obs = build_observation(
            row, stam,
            corpus_version="crop_supervised_v1.1",
            manifest_checksum="abc123",
        )
        pre.fit([obs], extractor=boom_extractor)
        sample = pre.transform(obs, extractor=boom_extractor)
        assert sample["ndvi"].shape[1:] == (1, pre.config.image.size, pre.config.image.size)
        assert sample["evi"].shape == sample["ndvi"].shape

    def test_tabular_features_vary_per_row(self) -> None:
        """ERROR 3: tabular vectors must differ between distinct rows."""
        row_a = _minimal_row("Belthangady", "coconut", 4)
        row_b = dict(_minimal_row("Puttur", "pepper", 6))
        row_b["lat"] = "13.100"
        row_b["lon"] = "75.900"
        row_b["spatial_match_distance_km"] = "2.3"
        row_b["season"] = "Rabi"

        stam = MagicMock()
        stam.resolve_sequence.return_value = _real_pair_sequence()
        from training.preprocessing.master_pipeline import Preprocessor
        import numpy as np
        from training.stam.patch_generator import RasterPatch

        def extractor(path, lon, lat, size=224):
            return RasterPatch(
                path=path,
                array=np.full((size, size), 0.42, dtype="float32"),
                mask=np.ones((size, size), dtype=bool),
                requested_size=size,
                window=(0, 0, size, size),
                bounds=(lon - 1, lat - 1, lon + 1, lat + 1),
                crs="EPSG:4326",
            )

        pre = Preprocessor.from_config(
            Path(__file__).resolve().parents[3] / "training" / "config" / "preprocessing.yaml"
        )
        objs = [
            build_observation(
                r, stam,
                corpus_version="crop_supervised_v1.1",
                manifest_checksum="abc123",
            )
            for r in (row_a, row_b)
        ]
        pre.fit(objs, extractor=extractor)
        va = pre.transform(objs[0], extractor=extractor)["tabular"]
        vb = pre.transform(objs[1], extractor=extractor)["tabular"]
        assert int(va.numel()) > 0, "tabular branch must be non-empty"
        assert not bool((va == vb).all().item()), "tabular vector must not be constant across rows"
