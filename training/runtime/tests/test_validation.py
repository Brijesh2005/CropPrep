"""Release validation tests (Phase R6)."""

from __future__ import annotations

import json

import pytest

from training.runtime import ReleaseLayout, ReleaseValidator
from training.runtime.exceptions import ReleaseValidationError
from training.runtime.tests.conftest import clone_release


@pytest.fixture
def release_validator():
    return ReleaseValidator()


def test_valid_release_all_checks(release_env, release_validator):
    result = release_validator.validate_release(release_env.release_path, strict=False)
    assert result.valid is True
    assert result.version == "1.0.0"
    assert set(result.checks) == {
        "integrity",
        "manifest",
        "versions",
        "dependencies",
        "compatibility",
        "smoke_test",
    }
    assert all(result.checks.values())
    assert result.errors == []
    assert result.backend == "pytorch"


def test_corrupted_model_fails_integrity(release_env, tmp_path, release_validator):
    bad = tmp_path / "bad_release"
    clone_release(release_env.release_path, bad)
    model = bad / "model" / "cropfusion.pt"
    model.write_bytes(model.read_bytes() + b"x")
    result = release_validator.validate_release(bad, strict=False)
    assert result.valid is False
    assert result.checks["integrity"] is False
    assert any("checksum mismatch" in error for error in result.errors)


def test_missing_required_file(release_env, tmp_path, release_validator):
    bad = tmp_path / "bad_release"
    clone_release(release_env.release_path, bad)
    (bad / "configs" / "model_config.yaml").unlink()
    result = release_validator.validate_release(bad, strict=False)
    assert result.valid is False
    assert result.checks["integrity"] is False
    assert any("model_config.yaml" in error for error in result.errors)


def test_tampered_manifest_schema(release_env, tmp_path, release_validator):
    bad = tmp_path / "bad_release"
    clone_release(release_env.release_path, bad)
    manifest_path = bad / "version" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["manifest_version"] = 99
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = release_validator.validate_release(bad, strict=False)
    assert result.checks["manifest"] is False
    assert any("manifest_version" in error for error in result.errors)


def test_inconsistent_versions(release_env, tmp_path, release_validator):
    bad = tmp_path / "bad_release"
    clone_release(release_env.release_path, bad)
    manifest_path = bad / "version" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["package_version"] = "2.0.0"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = release_validator.validate_release(bad, strict=False)
    assert result.checks["versions"] is False
    assert any("package_version" in error for error in result.errors)


def test_missing_checksums_file(release_env, tmp_path, release_validator):
    bad = tmp_path / "bad_release"
    clone_release(release_env.release_path, bad)
    (bad / "version" / "checksums.json").unlink()
    result = release_validator.validate_release(bad, strict=False)
    assert result.checks["integrity"] is False
    assert any("checksums.json is missing" in error for error in result.errors)


def test_strict_mode_raises(release_env, tmp_path, release_validator):
    bad = tmp_path / "bad_release"
    clone_release(release_env.release_path, bad)
    (bad / "model" / "cropfusion.pt").unlink()
    with pytest.raises(ReleaseValidationError) as exc_info:
        release_validator.validate_release(bad)
    assert "failed validation" in str(exc_info.value)
    detail = exc_info.value.detail
    assert detail["valid"] is False


def test_dependency_failure(release_env, release_validator, monkeypatch):
    layout = ReleaseLayout(release_env.release_path)

    from training.runtime import validation as validation_module

    def fake_importable(module):
        return module != "numpy"

    monkeypatch.setattr(validation_module, "_importable", fake_importable)
    ok, errors, _warnings = release_validator.verify_dependencies(layout)
    assert ok is False
    assert any("numpy" in error for error in errors)


def test_onnx_format_requires_onnxruntime(release_env, tmp_path, monkeypatch):
    import shutil

    from training.models import ModelExporter

    from training.runtime import validation as validation_module

    bad = tmp_path / "onnx_release"
    shutil.copytree(release_env.release_path, bad)
    ModelExporter(release_env.model).export_onnx(
        bad / "model" / "cropfusion.onnx"
    )
    layout = ReleaseLayout(bad)
    monkeypatch.setattr(
        validation_module, "_importable", lambda module: module != "onnxruntime"
    )
    ok, errors, _warnings = release_validator.verify_dependencies(layout)
    assert ok is False
    assert any("onnxruntime" in error for error in errors)


def test_validate_version_resolves(release_env, release_validator):
    result = release_validator.validate_version(
        release_env.root, "1.0.0", strict=False
    )
    assert result.valid is True
    assert result.version == "1.0.0"


def test_validate_version_missing(release_env, tmp_path, release_validator):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(Exception):
        release_validator.validate_version(empty, "9.9.9", strict=False)
