"""Versioning tests (Phase R5)."""

from __future__ import annotations

import pytest

from training.inference.versioning import (
    ResolvedVersions,
    build_version_files,
    bump_semver,
    content_sha256,
    file_sha256,
    model_fingerprint,
    resolve_versions,
    training_version_fingerprint,
)
from training.inference.exceptions import VersioningError


def test_content_sha256_deterministic():
    assert content_sha256({"a": 1}) == content_sha256({"a": 1})
    assert content_sha256({"a": 1}) != content_sha256({"a": 2})


def test_content_sha256_rejects_non_serializable():
    with pytest.raises(VersioningError):
        content_sha256({"x": object()})


def test_file_sha256(tmp_path):
    import hashlib

    path = tmp_path / "f.txt"
    path.write_text("hello", encoding="utf-8")
    assert file_sha256(path) == hashlib.sha256(b"hello").hexdigest()


def test_model_fingerprint_stable(model):
    fp1 = model_fingerprint(model)
    fp2 = model_fingerprint(model)
    assert fp1 == fp2
    assert len(fp1) == 64


def test_resolve_versions_defaults():
    resolved = resolve_versions(model_config_version="2.0.0")
    assert resolved.model_version == "2.0.0"
    assert resolved.dataset_version == "1.0.0"


def test_resolve_versions_overrides():
    resolved = resolve_versions(
        model_version="3.1.0",
        dataset_version="4.2.1",
        model_config_version="2.0.0",
    )
    assert resolved.model_version == "3.1.0"
    assert resolved.dataset_version == "4.2.1"


def test_invalid_version_raises():
    with pytest.raises(VersioningError):
        resolve_versions(package_version="not-semver")


def test_bump_semver():
    assert bump_semver("1.2.3", "major") == "2.0.0"
    assert bump_semver("1.2.3", "minor") == "1.3.0"
    assert bump_semver("1.2.3", "patch") == "1.2.4"


def test_build_version_files():
    resolved = ResolvedVersions(
        package_version="1.0.0",
        model_version="1.0.0",
        dataset_version="1.0.0",
    )
    dataset, model = build_version_files(
        resolved,
        model_fingerprint="a" * 64,
        dataset_fingerprint="b" * 64,
        training_fingerprint="c" * 64,
        git_commit_sha="deadbeef",
    )
    assert dataset["kind"] == "dataset"
    assert dataset["checksum"] == "b" * 64
    assert model["kind"] == "model"
    assert model["training_run_version"] == training_version_fingerprint(
        resolved, "c" * 64
    )
    assert model["git_commit"] == "deadbeef"
