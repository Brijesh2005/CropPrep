"""Tests for the training-data contract (R5.2.1 Task D)."""

from __future__ import annotations

import pytest

from training.preprocessing import (
    TrainingDataContract,
    assess_training_data_contract,
    validate_training_data_contract,
)
from training.preprocessing.data_contract import infer_yield_unit
from training.preprocessing.exceptions import DataContractViolationError
from training.stam.observation import AgriculturalObservation


def _obs(
    crop=None,
    yield_value=None,
    source_path="DK_Features_2020.csv",
    matched_level="district",
    paired=True,
):
    from training.stam.observation import (
        AdminLocation,
        GeographicPoint,
        ImagePairRef,
        ImageRecordRef,
        LocationInfo,
        QualityReport,
        SequenceInfo,
        TabularFeatures,
        TemporalInfo,
    )

    import uuid
    from datetime import date, datetime

    pair = None
    if paired:
        rec = ImageRecordRef(
            path=f"/data/{source_path}",
            relative_path=source_path,
            index_type="NDVI",
            resolution="R10m",
        )
        pair = ImagePairRef(date=date(2020, 7, 1), ndvi=rec, evi=rec)
    return AgriculturalObservation(
        observation_id=uuid.uuid4(),
        created_at=datetime(2020, 1, 1),
        location=LocationInfo(
            lon=74.8, lat=13.1,
            admin=AdminLocation(village="V" if matched_level == "village" else None,
                                district="DK", level=matched_level),
        ),
        temporal=TemporalInfo(year=2020, season="Kharif"),
        tabular=TabularFeatures(
            crop=crop,
            yield_value=yield_value,
            source_path=source_path,
            matched_level=matched_level,
        ),
        sequence=SequenceInfo(pairs=[pair] if pair else []),
        quality=QualityReport(passed=True, overall_score=90.0, issues=[]),
        crop=crop,
        yield_value=yield_value,
        patch_size=32,
    )


def _village_obs(crop="Coconut", yield_value=5200.0):
    return _obs(
        crop=crop,
        yield_value=yield_value,
        source_path="data_season.csv",
        matched_level="village",
    )


# --------------------------------------------------------------------------- #
# Unit inference
# --------------------------------------------------------------------------- #


def test_infer_yield_unit_kg_ha_from_data_season():
    assert infer_yield_unit("data_season.csv", "village", 5200.0) == "kg/ha"


def test_infer_yield_unit_npp_from_dk_features():
    assert infer_yield_unit("DK_Features_2020.csv", "district", 1.2) == "npp_proxy"


def test_infer_yield_unit_icrisat_kg_ha():
    assert infer_yield_unit("ICRISAT-District Level Data.csv", "district", 4000.0) == "kg/ha"


def test_infer_yield_unit_magnitude_fallback():
    assert infer_yield_unit(None, "none", 5200.0) == "kg/ha"
    assert infer_yield_unit(None, "none", 1.2) == "npp_proxy"
    assert infer_yield_unit(None, "none", None) is None


# --------------------------------------------------------------------------- #
# Assessment / validation
# --------------------------------------------------------------------------- #


def test_valid_homogeneous_village_corpus():
    corpus = [_village_obs("Coconut", 5200.0) for _ in range(10)]
    report = assess_training_data_contract(corpus)
    assert report.valid
    assert report.crop_training_samples == 10
    assert report.yield_training_samples == 10
    assert report.yield_unit == "kg/ha"
    assert "data_season.csv" in report.yield_source
    assert report.image_samples == 10
    assert report.tabular_samples == 10


def test_mixed_yield_units_rejected():
    corpus = [_village_obs("Coconut", 5200.0) for _ in range(5)]
    corpus += [_obs(crop="Rice", yield_value=1.2) for _ in range(5)]
    report = assess_training_data_contract(corpus)
    assert not report.valid
    assert any("mix" in e.lower() for e in report.errors)
    with pytest.raises(DataContractViolationError):
        validate_training_data_contract(corpus)


def test_non_strict_reports_but_does_not_raise():
    corpus = [_village_obs("Coconut", 5200.0) for _ in range(5)]
    corpus += [_obs(crop="Rice", yield_value=1.2) for _ in range(5)]
    report = validate_training_data_contract(corpus, strict=False)
    assert not report.valid


def test_crop_head_without_labels_rejected():
    corpus = [_obs(crop=None, yield_value=5200.0) for _ in range(5)]
    report = assess_training_data_contract(corpus)
    assert not report.valid
    assert any("crop label" in e.lower() for e in report.errors)
    with pytest.raises(DataContractViolationError):
        validate_training_data_contract(corpus)


def test_crop_head_disabled_ok_without_labels():
    corpus = [_obs(crop=None, yield_value=5200.0) for _ in range(5)]
    report = assess_training_data_contract(corpus, crop_head_enabled=False)
    assert report.valid
    assert report.crop_training_samples == 0


def test_partially_labeled_crop_warns_not_errors():
    corpus = [_village_obs("Coconut", 5200.0) for _ in range(8)]
    corpus += [_obs(crop=None, yield_value=1.2) for _ in range(2)]
    report = assess_training_data_contract(corpus)
    # Mixed units still make it invalid, but the label-warning path is exercised.
    assert any("unlabelled" in w.lower() for w in report.warnings)


def test_empty_corpus_invalid():
    report = assess_training_data_contract([])
    assert not report.valid
    assert any("empty" in e.lower() for e in report.errors)


def test_report_to_dict_fields():
    corpus = [_village_obs("Coconut", 5200.0) for _ in range(3)]
    d = assess_training_data_contract(corpus).to_dict()
    for key in (
        "crop_training_samples", "yield_training_samples", "yield_unit",
        "yield_source", "image_samples", "tabular_samples", "valid",
        "errors", "warnings",
    ):
        assert key in d


def test_validate_returns_report_on_valid():
    corpus = [_village_obs("Coconut", 5200.0) for _ in range(3)]
    report = validate_training_data_contract(corpus)
    assert isinstance(report, TrainingDataContract)
    assert report.valid
