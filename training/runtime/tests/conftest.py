"""Shared fixtures for the runtime test-suite (Phase R6)."""

from __future__ import annotations

import shutil
from types import SimpleNamespace

import pandas as pd
import pytest

from training.inference import (
    DatasetSources,
    InferenceConfig,
    InferencePackageBuilder,
)
from training.models import ModelConfig, ModelFactory
from training.preprocessing.label_pipeline import LabelPipeline
from training.preprocessing.tabular_pipeline import TabularPipeline
from training.preprocessing.transforms import LabelEncoder
from training.runtime import ReleasePackager


def small_config() -> ModelConfig:
    return ModelConfig(
        tabular={"numeric_dim": 3, "categorical_cardinalities": [2]},
        image_encoder={"backbone": "mobilenetv3_small_050", "input_size": 32},
        temporal={"d_model": 32, "depth": 1, "num_heads": 4, "ff_dim": 128,
                  "embedding_dim": 32, "max_len": 6},
        cross_attention={"num_heads": 4, "out_dim": 32},
        gated_fusion={"out_dim": 32, "hidden_dim": 32},
        shared_encoder={"d_model": 32, "depth": 1, "num_heads": 4, "ff_dim": 128,
                        "out_dim": 48},
        heads={"crop": {"num_classes": 3}, "yield_prediction": {}},
    )


def fake_preprocessor():
    """A Preprocessor-shaped object exposing ``tabular`` / ``label``."""
    tabular = TabularPipeline()
    tabular.numeric_features = ["rainfall_mm", "temperature", "soil_moisture"]
    tabular.categorical_features = ["soil_type"]
    tabular.feature_names = [
        "rainfall_mm", "temperature", "soil_moisture", "soil_type"
    ]
    tabular.fitted = True
    tabular._categorical_columns = ["soil_type"]

    label = LabelPipeline()
    label.crop_encoder = LabelEncoder().fit(["maize", "wheat", "rice"])
    label.fitted = True
    return SimpleNamespace(tabular=tabular, label=label)


def make_dataset_sources(tmp_path):
    src = tmp_path / "sources"
    src.mkdir(exist_ok=True)
    (src / "metadata.db").write_bytes(b"SQLITE")
    pd.DataFrame({"season": ["Kharif"]}).to_parquet(
        src / "historical_context.parquet"
    )
    pd.DataFrame({"lon": [74.8], "lat": [13.0]}).to_parquet(
        src / "location_index.parquet"
    )
    return DatasetSources(
        metadata_db=src / "metadata.db",
        historical_context=src / "historical_context.parquet",
        location_index=src / "location_index.parquet",
    )


def build_inference_package(base: "tmp_path"):
    """Build a Phase R5 inference package (pytorch format) under ``base``."""
    flat = base / "inference_package"
    cfg = InferenceConfig(
        general={"output_dir": str(flat)},
        exporter={"formats": ["pytorch"]},
        validation={"verify_checksums": True, "smoke_test": False},
    )
    model = ModelFactory.create(small_config())
    model.eval()
    preprocessor = fake_preprocessor()
    sources = make_dataset_sources(base)
    builder = InferencePackageBuilder(model, preprocessor, cfg)
    report = builder.build(
        sources,
        metrics={"crop": {"accuracy": 0.95}, "yield": {"rmse": 0.5}},
    )
    return flat, model, report


def build_release(base: "tmp_path", flat, version: str = "1.0.0"):
    """Pack the flat inference package into a release under ``base``."""
    target = base / f"cropfusion_release-v{version}"
    report = ReleasePackager().build(flat, target_dir=target)
    return target, report


def clone_release(release_path, target_path) -> None:
    """Copy a release directory (used before destructive test edits)."""
    shutil.copytree(release_path, target_path)


def refresh_checksums(release_path) -> None:
    """Recompute ``version/checksums.json`` after editing release files.

    Mirrors the packager: ``version/manifest.json``, ``version/checksums.json``
    and ``version/release_version.json`` are not part of the checksum map.
    """
    import json

    from shared.utils.hash import sha256_file

    root = release_path
    checksums = {}
    excluded = {
        "version/checksums.json",
        "version/manifest.json",
        "version/release_version.json",
    }
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if rel in excluded:
            continue
        checksums[rel] = sha256_file(path)
    out = root / "version" / "checksums.json"
    out.write_text(json.dumps(checksums, indent=2, sort_keys=True), encoding="utf-8")


@pytest.fixture(scope="module")
def release_env(tmp_path_factory):
    """One fully built release for the module."""
    root = tmp_path_factory.mktemp("runtime_env")
    flat, model, build_report = build_inference_package(root)
    release_path, release_report = build_release(root, flat)
    return SimpleNamespace(
        root=root,
        flat_dir=flat,
        model=model,
        build_report=build_report,
        release_path=release_path,
        release_report=release_report,
        version="1.0.0",
    )


@pytest.fixture
def releases_root(release_env):
    """The releases root containing the built release (module-scoped)."""
    return release_env.root
