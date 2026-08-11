"""Tests for training/kaggle/scripts/build_release.py.

The release-package builder is exercised end-to-end as a subprocess against
fixture files. It mirrors the app's ``cropfusion_release/`` contract: 12
required files plus a generated manifest and per-file sha256 checksum.
"""

from __future__ import annotations

import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_RELEASE = REPO_ROOT / "training" / "kaggle" / "scripts" / "build_release.py"

REQUIRED = [
    "model/cropfusion.pt",
    "metadata/metadata.db",
    "metadata/historical_context.parquet",
    "metadata/location_index.parquet",
    "metadata/village_metadata.parquet",
    "preprocess/scaler.pkl",
    "preprocess/label_encoder.pkl",
    "configs/model.yaml",
    "configs/inference.yaml",
    "version/manifest.json",
    "version/checksum.json",
    "reports/metrics.json",
]


def _write(path: Path, content: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")


def _make_sources(tmp_path: Path) -> Path:
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    src = tmp_path / "sources"
    feature_order = ["rainfall", "temperature", "area", "humidity", "price"]

    scaler = StandardScaler()
    scaler.mean_ = np.asarray([10.0, 25.0, 100.0, 60.0, 30.0], dtype="float64")
    scaler.scale_ = np.asarray([5.0, 2.0, 50.0, 10.0, 5.0], dtype="float64")
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(feature_order)
    scaler.feature_names = list(feature_order)
    _write(src / "preprocess" / "scaler.pkl", pickle.dumps(scaler))

    encoder = LabelEncoder()
    encoder.classes_ = np.asarray(["rice", "wheat", "maize"], dtype=object)
    _write(src / "preprocess" / "label_encoder.pkl", pickle.dumps(encoder))

    _write(src / "metadata" / "metadata.db", b"sqlite-bytes")
    index = pd.DataFrame(
        {
            "village": ["A", "A", "B"],
            "district": ["D1", "D1", "D2"],
            "taluk": ["T1", "T1", "T2"],
            "lon": [74.5, 74.5, 75.1],
            "lat": [15.2, 15.2, 16.0],
        }
    )
    index.to_parquet(src / "metadata" / "location_index.parquet", index=False)
    index.drop_duplicates(subset=["village", "district"]).to_parquet(
        src / "metadata" / "village_metadata.parquet", index=False
    )
    pd.DataFrame({"year": [2020, 2021], "record_count": [10, 12]}).to_parquet(
        src / "metadata" / "historical_context.parquet", index=False
    )

    _write(
        src / "reports" / "metrics.json",
        json.dumps({"val_loss": 0.5, "val_accuracy": 0.9}),
    )
    _write(
        src / "sources.json",
        json.dumps(
            {
                "model_version": "2.0.0",
                "dataset_version": "1.0.0",
                "feature_order": feature_order,
                "files": ["preprocess/scaler.pkl", "preprocess/label_encoder.pkl"],
            }
        ),
    )
    return src


def _make_exports(tmp_path: Path) -> tuple[Path, Path]:
    ts = tmp_path / "model.torchscript.pt"
    _write(ts, b"torchscript-bytes")
    cfg = tmp_path / "model.yaml"
    _write(
        cfg,
        "name: cropfusion_v1\nversion: 2.0.0\narchitecture_version: 1.0.0\n",
    )
    return ts, cfg


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(  # noqa: S603 - runs our own script with the test interpreter
        [sys.executable, str(BUILD_RELEASE), *args],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )


@pytest.fixture
def complete_fixture(tmp_path: Path):
    ts, cfg = _make_exports(tmp_path)
    sources = _make_sources(tmp_path)
    out = tmp_path / "releases"
    return ts, cfg, sources, out


def test_build_release_creates_valid_package(complete_fixture):
    ts, cfg, sources, out = complete_fixture
    result = _run(
        "--repo-root", str(REPO_ROOT),
        "--torchscript", str(ts),
        "--model-config", str(cfg),
        "--sources-dir", str(sources),
        "--output", str(out),
    )
    assert result.returncode == 0, result.stderr

    pkg = out / "cropfusion_release-v2.0.0"
    for rel in REQUIRED:
        assert (pkg / rel).exists(), f"missing {rel}"

    manifest = json.loads((pkg / "version" / "manifest.json").read_text())
    assert manifest["format"] == "cropfusion_release"
    assert manifest["model_version"] == "2.0.0"
    assert manifest["dataset_version"] == "1.0.0"
    assert set(manifest["files"]) == set(REQUIRED) - {"version/manifest.json"}

    checksums = json.loads((pkg / "version" / "checksum.json").read_text())
    assert set(checksums["files"]) == (
        set(REQUIRED) - {"version/checksum.json", "version/manifest.json"}
    )
    assert all(len(hexdigest) == 64
               for hexdigest in checksums["files"].values())

    model_yaml = (pkg / "configs" / "model.yaml").read_text()
    assert "feature_order" in model_yaml
    assert "input_dim: 5" in model_yaml


def test_build_release_infers_feature_order_from_scaler_without_sources_json(
    complete_fixture,
):
    ts, cfg, sources, out = complete_fixture
    (sources / "sources.json").unlink()
    result = _run(
        "--repo-root", str(REPO_ROOT),
        "--torchscript", str(ts),
        "--model-config", str(cfg),
        "--sources-dir", str(sources),
        "--output", str(out),
    )
    assert result.returncode == 0, result.stderr
    model_yaml = (out / "cropfusion_release-v2.0.0" / "configs"
                  / "model.yaml").read_text()
    assert "feature_order" in model_yaml


def test_build_release_missing_sources_fails(complete_fixture):
    ts, cfg, _sources, out = complete_fixture
    result = _run(
        "--repo-root", str(REPO_ROOT),
        "--torchscript", str(ts),
        "--model-config", str(cfg),
        "--output", str(out),
    )
    assert result.returncode != 0
    assert "release package incomplete" in result.stderr
    for missing in ("preprocess/scaler.pkl", "metadata/metadata.db"):
        assert missing in result.stderr


def test_build_release_allow_partial(complete_fixture):
    ts, cfg, _sources, out = complete_fixture
    result = _run(
        "--repo-root", str(REPO_ROOT),
        "--torchscript", str(ts),
        "--model-config", str(cfg),
        "--output", str(out),
        "--allow-partial",
    )
    assert result.returncode == 0, result.stderr
    assert "WARNING" in result.stdout
    pkg = out / "cropfusion_release-v2.0.0"
    assert (pkg / "version" / "manifest.json").exists()


def test_build_release_explicit_version_wins(complete_fixture):
    ts, cfg, sources, out = complete_fixture
    result = _run(
        "--repo-root", str(REPO_ROOT),
        "--torchscript", str(ts),
        "--model-config", str(cfg),
        "--sources-dir", str(sources),
        "--output", str(out),
        "--version", "9.9.9",
    )
    assert result.returncode == 0, result.stderr
    assert (out / "cropfusion_release-v9.9.9" / "version"
            / "manifest.json").exists()
