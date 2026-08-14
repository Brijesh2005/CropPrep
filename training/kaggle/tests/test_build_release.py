"""build_release.py -> Prediction Platform contract tests.

``package_sources.py`` (train kernel) + ``build_release.py`` (export kernel)
must assemble the exact ``cropfusion_release/`` tree the app-side
``ReleasePackageLoader`` validates. These tests pin that contract: every
required file the loader demands must be produced, and the emitted package
must pass the loader's files-exist and checksum validation.
"""

from __future__ import annotations

import importlib
import json
import pickle
import sys
from pathlib import Path

from training.kaggle.scripts.build_release import REQUIRED_RELEASE_FILES, build_release

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _import_app(mod: str):
    """Import an ``application/`` module via the layout the app itself uses.

    The app package lives under ``application/`` and imports it internally as
    ``inference_package.release.*`` (with ``application/`` on sys.path), so we
    mirror that rather than ``application.inference_package.*``.
    """
    app_dir = str(_REPO_ROOT / "application")
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)
    return importlib.import_module(mod)


def _write_bytes(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, bytes):
        path.write_bytes(payload)
    else:
        path.write_text(str(payload), encoding="utf-8")


def _make_sources(tmp_path: Path) -> Path:
    sources = tmp_path / "sources"
    for rel, payload in (
        ("preprocess/scaler.pkl", pickle.dumps({"scaler": 1})),
        ("preprocess/label_encoder.pkl", pickle.dumps({"encoder": 1})),
        ("metadata/metadata.db", b"sqlite-snapshot"),
        ("metadata/historical_context.parquet", b"parquet"),
        ("metadata/location_index.parquet", b"parquet"),
        ("metadata/village_metadata.parquet", b"parquet"),
        ("reports/metrics.json", json.dumps({"model_version": "1.0.0"})),
    ):
        _write_bytes(sources / rel, payload)
    (sources / "sources.json").write_text(
        json.dumps({"feature_order": ["a", "b"], "dataset_version": "2.0.0"}),
        encoding="utf-8",
    )
    return sources


def test_required_files_match_app_loader_contract() -> None:
    manifest = _import_app("inference_package.release.manifest")
    app_required = {
        artifact.rel_path
        for artifact in manifest.RELEASE_PACKAGE_FILES
        if artifact.required
    }
    assert set(REQUIRED_RELEASE_FILES) == app_required
    assert len(REQUIRED_RELEASE_FILES) == 12


def test_build_release_produces_all_required_files(tmp_path: Path) -> None:
    sources = _make_sources(tmp_path)
    torchscript = tmp_path / "model.torchscript.pt"
    model_yaml = tmp_path / "model.yaml"
    _write_bytes(torchscript, b"torchscript-payload")
    model_yaml.write_text("name: cropfusion\nversion: 1.0.0\n", encoding="utf-8")
    output = tmp_path / "releases"

    report = build_release(
        torchscript=torchscript,
        model_config_path=model_yaml,
        sources_dir=sources,
        output_dir=output,
    )

    assert report["valid"] is True
    assert report["missing_required"] == []
    release_dir = Path(report["release_dir"])
    for rel in REQUIRED_RELEASE_FILES:
        assert (release_dir / rel).exists(), f"missing {rel}"


def test_build_release_package_passes_loader_validation(tmp_path: Path) -> None:
    loader_mod = _import_app("inference_package.release.loader")

    sources = _make_sources(tmp_path)
    torchscript = tmp_path / "model.torchscript.pt"
    model_yaml = tmp_path / "model.yaml"
    _write_bytes(torchscript, b"torchscript-payload")
    model_yaml.write_text("name: cropfusion\nversion: 1.0.0\n", encoding="utf-8")
    output = tmp_path / "releases"

    report = build_release(
        torchscript=torchscript,
        model_config_path=model_yaml,
        sources_dir=sources,
        output_dir=output,
    )

    loader = loader_mod.ReleasePackageLoader(Path(report["release_dir"]))
    loader._validate_files_exist()
    loader._validate_checksums(loader._load_json("version/checksum.json"))


def test_build_release_missing_source_reports_incomplete(tmp_path: Path) -> None:
    sources = _make_sources(tmp_path)
    (sources / "preprocess" / "scaler.pkl").unlink()
    torchscript = tmp_path / "model.torchscript.pt"
    model_yaml = tmp_path / "model.yaml"
    _write_bytes(torchscript, b"torchscript-payload")
    model_yaml.write_text("name: cropfusion\nversion: 1.0.0\n", encoding="utf-8")

    report = build_release(
        torchscript=torchscript,
        model_config_path=model_yaml,
        sources_dir=sources,
        output_dir=tmp_path / "releases",
        allow_partial=True,
    )

    assert report["valid"] is False
    assert "preprocess/scaler.pkl" in report["missing_required"]
