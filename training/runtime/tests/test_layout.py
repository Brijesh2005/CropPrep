"""Layout + packager tests (Phase R6)."""

from __future__ import annotations

import pytest

from training.runtime import (
    RELEASE_DIRS,
    REQUIRED_RELEASE_FILES,
    ReleaseLayout,
    ReleaseManifest,
    ReleasePackager,
    parse_release_dir,
    release_dir_name,
)
from training.runtime.exceptions import (
    ReleaseLayoutError,
    ReleasePackagingError,
)
from training.runtime.layout import iter_release_dirs, resolve_release


# --------------------------------------------------------------------------- #
# Naming / parsing
# --------------------------------------------------------------------------- #

def test_parse_release_dir():
    assert parse_release_dir("cropfusion_release-v1.0.0") == "1.0.0"
    assert parse_release_dir("cropfusion_release_v1.2.3") == "1.2.3"
    assert parse_release_dir("cropfusion_release") is None
    assert parse_release_dir("something_else") is None
    assert parse_release_dir("cropfusion_release-v1.0") is None


def test_release_dir_name():
    assert release_dir_name("1.0.0") == "cropfusion_release-v1.0.0"
    with pytest.raises(ReleaseLayoutError):
        release_dir_name("nope")


# --------------------------------------------------------------------------- #
# Layout resolution
# --------------------------------------------------------------------------- #

def test_layout_paths(release_env):
    layout = ReleaseLayout(release_env.release_path)
    assert layout.root == release_env.release_path
    assert layout.model_dir.name == "model"
    assert layout.preprocess_dir.name == "preprocess"
    assert layout.metadata_dir.name == "metadata"
    assert layout.configs_dir.name == "configs"
    assert layout.reports_dir.name == "reports"
    assert layout.version_dir.name == "version"
    assert layout.artifact("model/cropfusion.pt").name == "cropfusion.pt"


def test_layout_formats(release_env):
    layout = ReleaseLayout(release_env.release_path)
    assert layout.has_format("pytorch")
    assert layout.formats == ["pytorch"]


def test_layout_valid_structure(release_env):
    layout = ReleaseLayout(release_env.release_path)
    valid, errors = layout.is_valid_structure()
    assert valid, errors
    assert not errors


def test_layout_manifest_and_checksums(release_env):
    layout = ReleaseLayout(release_env.release_path)
    manifest = layout.manifest()
    assert manifest.manifest_version == 2
    assert manifest.package_version == "1.0.0"
    assert manifest.release_version == "1.0.0"
    assert "model/cropfusion.pt" in manifest.files
    checksums = layout.checksums()
    assert isinstance(checksums, dict)
    assert len(checksums) >= len(REQUIRED_RELEASE_FILES) - 2


def test_required_release_dirs_and_files():
    assert set(RELEASE_DIRS) == {
        "model", "preprocess", "metadata", "configs", "reports", "version"
    }
    assert "README.md" in REQUIRED_RELEASE_FILES
    assert "version/manifest.json" in REQUIRED_RELEASE_FILES
    assert "metadata/feature_lookup.parquet" in REQUIRED_RELEASE_FILES


# --------------------------------------------------------------------------- #
# Packager
# --------------------------------------------------------------------------- #

def test_packager_build(release_env):
    report = release_env.release_report
    assert report.version == "1.0.0"
    assert report.target_dir == release_env.release_path
    assert "model/cropfusion.pt" in report.files
    assert "version/manifest.json" in report.files
    assert "metadata/feature_lookup.parquet" in report.files
    assert "version/original_checksums.json" in report.files
    assert "version/original_manifest.json" in report.files
    assert report.manifest["package_version"] == "1.0.0"
    assert report.manifest["manifest_version"] == 2


def test_packager_generates_derived_artefacts(release_env):
    layout = ReleaseLayout(release_env.release_path)
    assert layout.exists("metadata/feature_lookup.parquet")
    assert layout.exists("model/model_metadata.json")
    assert layout.exists("preprocess/preprocess_metadata.json")
    assert layout.exists("README.md")

    lookup = layout.artifact("metadata/feature_lookup.parquet")
    import pandas as pd

    df = pd.read_parquet(lookup)
    assert list(df.columns) == ["feature_index", "feature_name", "feature_type", "feature_group"]
    names = list(df["feature_name"])
    assert "rainfall_mm" in names
    assert "soil_type" in names
    rows = df.to_dict("records")
    soil = [r for r in rows if r["feature_name"] == "soil_type"]
    assert soil and soil[0]["feature_type"] == "categorical"
    assert df["feature_index"].tolist() == list(range(len(df)))


def test_packager_missing_source(tmp_path):
    with pytest.raises(ReleasePackagingError):
        ReleasePackager().build(tmp_path / "missing")


def test_packager_incomplete_source(release_env, tmp_path):
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / "cropfusion.pt").write_bytes(b"x")
    with pytest.raises(ReleasePackagingError) as excinfo:
        ReleasePackager().build(incomplete, target_dir=tmp_path / "out")
    assert "missing" in str(excinfo.value).lower()


def test_packager_version_override(release_env, tmp_path):
    target = tmp_path / "cropfusion_release-v2.3.4"
    report = ReleasePackager().build(
        release_env.flat_dir, target_dir=target, version="2.3.4"
    )
    assert report.version == "2.3.4"
    assert ReleaseManifest.load(target).package_version == "2.3.4"


def test_packager_invalid_version(release_env, tmp_path):
    with pytest.raises(ReleasePackagingError):
        ReleasePackager().build(
            release_env.flat_dir, target_dir=tmp_path / "out", version="x.y.z"
        )


def test_packager_releases_root(release_env, tmp_path):
    root = tmp_path / "releases"
    report = ReleasePackager().build(release_env.flat_dir, releases_root=root)
    assert report.target_dir == root / "cropfusion_release-v1.0.0"
    found = iter_release_dirs(root)
    assert [(name, version) for name, _, version in found] == [
        ("cropfusion_release-v1.0.0", "1.0.0")
    ]
    assert resolve_release(root, "1.0.0") == report.target_dir


def test_packager_readme(release_env):
    text = (release_env.release_path / "README.md").read_text(encoding="utf-8")
    assert "release package" in text.lower()
    assert "model/" in text
