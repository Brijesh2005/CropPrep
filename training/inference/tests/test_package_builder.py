"""Inference package builder + validator tests (Phase R5)."""

from __future__ import annotations

import json

import pytest

from training.inference.config import InferenceConfig
from training.inference.exceptions import PackageValidationError
from training.inference.package_builder import (
    REQUIRED_ARTIFACTS,
    InferencePackageBuilder,
)
from training.inference.validate import InferencePackageValidator

from .conftest import make_dataset_sources


@pytest.fixture
def inference_config(tmp_path) -> InferenceConfig:
    return InferenceConfig(
        general={
            "output_dir": str(tmp_path / "package"),
            "package_name": "cropfusion-test",
            "version": "0.1.0",
            "model_version": "1.2.3",
        },
        exporter={"formats": ["pytorch"]},
        validation={"smoke_test": True, "strict": False},
    )


def _build(model, preprocessor, inference_config, tmp_path):
    sources = make_dataset_sources(tmp_path)
    builder = InferencePackageBuilder(model, preprocessor, inference_config)
    return builder.build(
        sources,
        metrics={"crop": {"accuracy": 0.9}},
        training_config={"epochs": 10, "batch_size": 32},
        dataset_manager=None,
    )


def test_required_artifacts():
    assert len(REQUIRED_ARTIFACTS) == 14
    assert "cropfusion.pt" in REQUIRED_ARTIFACTS
    assert "manifest.json" not in REQUIRED_ARTIFACTS


def test_build_produces_all_artifacts(model, preprocessor, inference_config, tmp_path):
    report = _build(model, preprocessor, inference_config, tmp_path)
    names = {p.name for p in report.files.values()}
    for artifact in REQUIRED_ARTIFACTS:
        assert artifact in names, artifact
    assert "manifest.json" in names
    assert (tmp_path / "package" / "cropfusion.pt").exists()
    assert (tmp_path / "package" / "feature_scalers.pkl").exists()
    assert (tmp_path / "package" / "label_encoder.pkl").exists()


def test_manifest_content(model, preprocessor, inference_config, tmp_path):
    report = _build(model, preprocessor, inference_config, tmp_path)
    manifest = report.manifest
    assert manifest["manifest_version"] == 1
    assert manifest["package_name"] == "cropfusion-test"
    assert manifest["package_version"] == "0.1.0"
    assert manifest["model_version"] == "1.2.3"
    assert manifest["dataset_version"] == "1.0.0"
    assert manifest["formats"] == ["pytorch"]
    assert manifest["training_fingerprint"]
    assert manifest["model_fingerprint"]
    assert set(manifest["files"]) == set(
        json.loads(
            (tmp_path / "package" / "checksums.json").read_text(encoding="utf-8")
        )
    )


def test_checksums_verify(model, preprocessor, inference_config, tmp_path):
    report = _build(model, preprocessor, inference_config, tmp_path)
    checksums = json.loads(
        (tmp_path / "package" / "checksums.json").read_text(encoding="utf-8")
    )
    assert "checksums.json" not in checksums  # self-referential exclusion
    validator = InferencePackageValidator(inference_config)
    result = validator.validate_package(tmp_path / "package")
    assert result.valid is True, result.errors
    assert result.checks["integrity"] is True
    assert result.checks["compatibility"] is True
    assert result.checks["smoke_test"] is True


def test_tamper_detected(model, preprocessor, inference_config, tmp_path):
    _build(model, preprocessor, inference_config, tmp_path)
    target = tmp_path / "package" / "cropfusion.pt"
    target.write_bytes(target.read_bytes() + b"\x00")
    result = InferencePackageValidator(inference_config).validate_package(
        tmp_path / "package"
    )
    assert result.valid is False
    assert any("checksum mismatch" in error for error in result.errors)


def test_strict_raises_on_invalid(model, preprocessor, inference_config, tmp_path):
    _build(model, preprocessor, inference_config, tmp_path)
    inference_config.validation.strict = True
    target = tmp_path / "package" / "cropfusion.pt"
    target.write_bytes(target.read_bytes() + b"\x00")
    with pytest.raises(PackageValidationError):
        InferencePackageValidator(inference_config).validate_package(
            tmp_path / "package"
        )


def test_build_report_to_dict(model, preprocessor, inference_config, tmp_path):
    report = _build(model, preprocessor, inference_config, tmp_path)
    payload = report.to_dict()
    assert payload["output_dir"].endswith("package")
    assert payload["validation"]["valid"] is True
