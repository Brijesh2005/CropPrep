"""Unit tests for scalers and encoders."""

from __future__ import annotations

import numpy as np
import pytest

from training.preprocessing.transforms import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)


def test_standard_scaler():
    X = np.asarray([[1.0], [2.0], [3.0], [4.0]])
    scaler = StandardScaler().fit(X)
    scaled = scaler.transform(X)
    assert np.allclose(scaled.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(scaled.std(axis=0), 1.0, atol=1e-6)
    # Round trip.
    assert np.allclose(scaler.inverse_transform(scaled), X, atol=1e-6)


def test_standard_scaler_handles_constant():
    X = np.asarray([[5.0], [5.0], [5.0]])
    scaled = StandardScaler().fit_transform(X)
    assert np.allclose(scaled, 0.0)  # no div-by-zero


def test_minmax_scaler():
    X = np.asarray([[0.0], [5.0], [10.0]])
    scaled = MinMaxScaler().fit_transform(X)
    assert np.allclose(scaled, [[0.0], [0.5], [1.0]])
    assert np.allclose(MinMaxScaler().fit(X).inverse_transform(scaled), X)


def test_robust_scaler():
    X = np.asarray([[1.0], [2.0], [3.0], [100.0]])
    scaler = RobustScaler().fit(X)
    # The median (2.5) maps to ~0 despite the outlier at 100.
    assert abs(float(scaler.transform([[2.5]])[0, 0])) < 1e-6
    # The median element (value 3) is far from the outlier effect.
    assert abs(float(scaler.transform([[3.0]])[0, 0])) < 1.0


def test_scaler_persistence(tmp_path):
    scaler = StandardScaler().fit(np.asarray([[1.0], [2.0], [3.0]]))
    path = scaler.save(tmp_path / "scaler.pkl")
    loaded = StandardScaler.load(path)
    assert np.allclose(loaded.transform([[2.5]]), scaler.transform([[2.5]]))


def test_ordinal_encoder():
    X = np.asarray([["b"], ["a"], ["b"]])
    encoder = OrdinalEncoder().fit(X)
    encoded = encoder.transform([["a"], ["c"]])
    # Categories preserve first-seen order: ["b", "a"] -> 'a' is index 1.
    assert encoded[0, 0] == 1
    assert encoded[1, 0] == -1  # unseen


def test_onehot_encoder():
    X = np.asarray([["red"], ["green"], ["red"]])
    encoder = OneHotEncoder().fit(X)
    encoded = encoder.transform([["green"]])
    assert encoded[0, 1] == 1.0  # 'green' second category


def test_label_encoder_roundtrip():
    encoder = LabelEncoder().fit(["Rice", "Coconut", "Rice"])
    codes = encoder.transform(["Rice", "Coconut"])
    assert list(codes) == [0, 1]
    assert encoder.inverse_transform([0, 1]) == ["Rice", "Coconut"]
    assert encoder.num_classes == 2


def test_label_encoder_persistence(tmp_path):
    encoder = LabelEncoder().fit(["a", "b", "c"])
    path = encoder.save(tmp_path / "label.pkl")
    loaded = LabelEncoder.load(path)
    assert loaded.transform(["b"])[0] == encoder.transform(["b"])[0]
