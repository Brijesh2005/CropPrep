"""Tests for training/kaggle/scripts/package_sources.py (pure helpers).

The full train-kernel flow needs the dataset manager + STAM, which is not
unit-testable here; these tests cover the sklearn serialization contract that
``build_release.py`` and the Prediction Platform loader depend on.
"""

from __future__ import annotations

import importlib.util
import pickle
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "training" / "kaggle" / "scripts" / "package_sources.py"

_spec = importlib.util.spec_from_file_location("package_sources", _SCRIPT)
ps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ps)


class _FakeScaler:
    mean_ = np.asarray([10.0, 25.0])
    scale_ = np.asarray([5.0, 2.0])


class _FakeTabular:
    scaler = _FakeScaler()
    numeric_features = ["rainfall", "temperature"]


class _FakeCropEncoder:
    classes_ = np.asarray(["rice", "wheat", "maize"], dtype=object)


class _FakeLabel:
    crop_encoder = _FakeCropEncoder()
    num_classes = 3


class _FakePreprocessor:
    tabular = _FakeTabular()
    label = _FakeLabel()


def test_to_sklearn_scaler_keeps_features_and_names():
    scaler = ps._to_sklearn_scaler(_FakeScaler(), ["rainfall", "temperature"])
    assert scaler.n_features_in_ == 2
    assert scaler.feature_names == ["rainfall", "temperature"]
    np.testing.assert_allclose(scaler.mean_, [10.0, 25.0])
    np.testing.assert_allclose(scaler.scale_, [5.0, 2.0])
    np.testing.assert_allclose(scaler.var_, [25.0, 4.0])


def test_to_sklearn_label_encoder_keeps_classes():
    encoder = ps._to_sklearn_label_encoder(_FakeCropEncoder())
    assert list(encoder.classes_) == ["rice", "wheat", "maize"]


def test_persist_pipeline_writes_sklearn_artefacts(tmp_path):
    meta = ps._persist_pipeline(_FakePreprocessor(), tmp_path)
    assert meta["feature_order"] == ["rainfall", "temperature"]
    assert meta["num_features"] == 2
    assert meta["num_classes"] == 3
    assert meta["crop_classes"] == ["rice", "wheat", "maize"]

    scaler = pickle.loads(  # noqa: S301 - round-trips our own fixture
        (tmp_path / "preprocess" / "scaler.pkl").read_bytes()
    )
    assert scaler.feature_names == ["rainfall", "temperature"]
    encoder = pickle.loads(  # noqa: S301 - round-trips our own fixture
        (tmp_path / "preprocess" / "label_encoder.pkl").read_bytes()
    )
    assert list(encoder.classes_) == ["rice", "wheat", "maize"]


def test_persist_pipeline_requires_fitted_scaler(tmp_path):
    class _NoScaler(_FakePreprocessor):
        class tabular(_FakeTabular):  # type: ignore[no-redef]
            scaler = None

    with pytest.raises(RuntimeError, match="no tabular scaler"):
        ps._persist_pipeline(_NoScaler(), tmp_path)
