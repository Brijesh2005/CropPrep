"""Fitted numeric transforms: scalers and encoders.

These are self-contained numpy implementations (no sklearn dependency) that
follow the :class:`~ai.preprocessing.interfaces.Transformer` contract so they
persist to / load from pickle and are reproducible across pipeline runs.

Supported:

* :class:`StandardScaler` — zero mean / unit variance.
* :class:`MinMaxScaler` — scale into ``[0, 1]``.
* :class:`RobustScaler` — centre by median, scale by IQR.
* :class:`OrdinalEncoder` — map categories to integer codes per column.
* :class:`OneHotEncoder` — one-hot encode categorical columns.
* :class:`LabelEncoder` — map a single label column to integer codes.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .exceptions import PreprocessingError
from .interfaces import Transformer
from .utils import is_numeric


def _as_float(X: Any) -> np.ndarray:
    return np.asarray(X, dtype="float64")


def _fill_nan_to_zero(array: np.ndarray) -> np.ndarray:
    return np.nan_to_num(array, nan=0.0)


# --------------------------------------------------------------------------- #
# Scalers
# --------------------------------------------------------------------------- #


class StandardScaler(Transformer):
    """Zero-mean, unit-variance scaling (per feature)."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.fitted = False

    def fit(self, X: Any) -> "StandardScaler":
        array = _as_float(X)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        self.mean_ = np.nanmean(array, axis=0)
        self.scale_ = np.nanstd(array, axis=0)
        self.scale_[self.scale_ == 0] = 1.0  # avoid div-by-zero on constants
        self.fitted = True
        return self

    def transform(self, X: Any) -> np.ndarray:
        self._require_fitted()
        array = _as_float(X)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        scaled = (array - self.mean_) / self.scale_
        return _fill_nan_to_zero(scaled)

    def fit_transform(self, X: Any) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: Any) -> np.ndarray:
        self._require_fitted()
        array = _as_float(X)
        return array * self.scale_ + self.mean_

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": "standard",
            "mean": self.mean_.tolist() if self.mean_ is not None else None,
            "scale": self.scale_.tolist() if self.scale_ is not None else None,
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "StandardScaler":
        scaler = cls()
        if state.get("mean") is not None:
            scaler.mean_ = np.asarray(state["mean"], dtype="float64")
            scaler.scale_ = np.asarray(state["scale"], dtype="float64")
            scaler.fitted = True
        return scaler

    def _require_fitted(self) -> None:
        if not self.fitted:
            raise PreprocessingError("StandardScaler is not fitted")


class MinMaxScaler(Transformer):
    """Scale features into ``[0, 1]`` using observed min/max."""

    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0)) -> None:
        self.feature_range = feature_range
        self.min_: np.ndarray | None = None
        self.max_: np.ndarray | None = None
        self.fitted = False

    def fit(self, X: Any) -> "MinMaxScaler":
        array = _as_float(X)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        self.min_ = np.nanmin(array, axis=0)
        self.max_ = np.nanmax(array, axis=0)
        span = self.max_ - self.min_
        span[span == 0] = 1.0
        self._span = span
        self.fitted = True
        return self

    def transform(self, X: Any) -> np.ndarray:
        self._require_fitted()
        array = _as_float(X)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        lo, hi = self.feature_range
        scaled = (array - self.min_) / self._span * (hi - lo) + lo
        return _fill_nan_to_zero(scaled)

    def fit_transform(self, X: Any) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: Any) -> np.ndarray:
        self._require_fitted()
        array = _as_float(X)
        lo, hi = self.feature_range
        return (array - lo) / (hi - lo) * self._span + self.min_

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": "minmax",
            "min": self.min_.tolist() if self.min_ is not None else None,
            "max": self.max_.tolist() if self.max_ is not None else None,
            "feature_range": list(self.feature_range),
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "MinMaxScaler":
        scaler = cls(feature_range=tuple(state.get("feature_range", (0.0, 1.0))))
        if state.get("min") is not None:
            scaler.min_ = np.asarray(state["min"], dtype="float64")
            scaler.max_ = np.asarray(state["max"], dtype="float64")
            span = scaler.max_ - scaler.min_
            span[span == 0] = 1.0
            scaler._span = span
            scaler.fitted = True
        return scaler

    def _require_fitted(self) -> None:
        if not self.fitted:
            raise PreprocessingError("MinMaxScaler is not fitted")


class RobustScaler(Transformer):
    """Median-centre, IQR-scale scaling (robust to outliers)."""

    def __init__(self) -> None:
        self.median_: np.ndarray | None = None
        self.iqr_: np.ndarray | None = None
        self.fitted = False

    def fit(self, X: Any) -> "RobustScaler":
        array = _as_float(X)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        q75 = np.nanpercentile(array, 75, axis=0)
        q25 = np.nanpercentile(array, 25, axis=0)
        self.median_ = np.nanmedian(array, axis=0)
        iqr = q75 - q25
        iqr[iqr == 0] = 1.0
        self.iqr_ = iqr
        self.fitted = True
        return self

    def transform(self, X: Any) -> np.ndarray:
        self._require_fitted()
        array = _as_float(X)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        scaled = (array - self.median_) / self.iqr_
        return _fill_nan_to_zero(scaled)

    def fit_transform(self, X: Any) -> np.ndarray:
        return self.fit(X).transform(X)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": "robust",
            "median": self.median_.tolist() if self.median_ is not None else None,
            "iqr": self.iqr_.tolist() if self.iqr_ is not None else None,
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "RobustScaler":
        scaler = cls()
        if state.get("median") is not None:
            scaler.median_ = np.asarray(state["median"], dtype="float64")
            scaler.iqr_ = np.asarray(state["iqr"], dtype="float64")
            scaler.fitted = True
        return scaler

    def _require_fitted(self) -> None:
        if not self.fitted:
            raise PreprocessingError("RobustScaler is not fitted")


# --------------------------------------------------------------------------- #
# Encoders
# --------------------------------------------------------------------------- #


class OrdinalEncoder(Transformer):
    """Map categorical values to integer codes per column."""

    def __init__(self) -> None:
        self.categories_: list[list[str]] = []
        self.fitted = False

    def fit(self, X: Any) -> "OrdinalEncoder":
        array = np.asarray(X)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        self.categories_ = []
        for col in range(array.shape[1]):
            unique = _ordered_unique(array[:, col])
            self.categories_.append(unique)
        self.fitted = True
        return self

    def transform(self, X: Any) -> np.ndarray:
        self._require_fitted()
        array = np.asarray(X)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.shape[1] != len(self.categories_):
            raise PreprocessingError(
                f"OrdinalEncoder expected {len(self.categories_)} columns, "
                f"got {array.shape[1]}"
            )
        out = np.zeros(array.shape, dtype="int64")
        for col, mapping in enumerate(self.categories_):
            index = {cat: i for i, cat in enumerate(mapping)}
            for row, value in enumerate(array[:, col]):
                out[row, col] = index.get(str(value), -1)  # -1 => unseen
        return out

    def fit_transform(self, X: Any) -> np.ndarray:
        return self.fit(X).transform(X)

    def to_dict(self) -> dict[str, Any]:
        return {"name": "ordinal", "categories": self.categories_}

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "OrdinalEncoder":
        encoder = cls()
        encoder.categories_ = [list(c) for c in state.get("categories", [])]
        encoder.fitted = bool(encoder.categories_)
        return encoder

    def _require_fitted(self) -> None:
        if not self.fitted:
            raise PreprocessingError("OrdinalEncoder is not fitted")


class OneHotEncoder(Transformer):
    """One-hot encode categorical columns (dense output)."""

    def __init__(self) -> None:
        self.categories_: list[list[str]] = []
        self.fitted = False

    def fit(self, X: Any) -> "OneHotEncoder":
        array = np.asarray(X)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        self.categories_ = []
        for col in range(array.shape[1]):
            self.categories_.append(_ordered_unique(array[:, col]))
        self.fitted = True
        return self

    def transform(self, X: Any) -> np.ndarray:
        self._require_fitted()
        array = np.asarray(X)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.shape[1] != len(self.categories_):
            raise PreprocessingError(
                f"OneHotEncoder expected {len(self.categories_)} columns, "
                f"got {array.shape[1]}"
            )
        rows: list[np.ndarray] = []
        for row in range(array.shape[0]):
            encoded: list[float] = []
            for col, mapping in enumerate(self.categories_):
                index = {cat: i for i, cat in enumerate(mapping)}
                code = index.get(str(array[row, col]), None)
                one = [0.0] * len(mapping)
                if code is not None:
                    one[code] = 1.0
                encoded.extend(one)
            rows.append(encoded)
        return np.asarray(rows, dtype="float32")

    def fit_transform(self, X: Any) -> np.ndarray:
        return self.fit(X).transform(X)

    @property
    def output_dimension(self) -> int:
        return sum(len(c) for c in self.categories_) if self.fitted else 0

    def to_dict(self) -> dict[str, Any]:
        return {"name": "onehot", "categories": self.categories_}

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "OneHotEncoder":
        encoder = cls()
        encoder.categories_ = [list(c) for c in state.get("categories", [])]
        encoder.fitted = bool(encoder.categories_)
        return encoder

    def _require_fitted(self) -> None:
        if not self.fitted:
            raise PreprocessingError("OneHotEncoder is not fitted")


class LabelEncoder(Transformer):
    """Encode a single label column into integer codes (0..N-1)."""

    def __init__(self) -> None:
        self.classes_: list[str] = []
        self.fitted = False

    def fit(self, X: Any) -> "LabelEncoder":
        values = [str(v) for v in np.asarray(X).ravel() if v is not None]
        self.classes_ = _ordered_unique(values)
        self.fitted = True
        return self

    def transform(self, X: Any) -> np.ndarray:
        self._require_fitted()
        index = {cls: i for i, cls in enumerate(self.classes_)}
        values = [str(v) for v in np.asarray(X).ravel()]
        return np.asarray([index.get(v, -1) for v in values], dtype="int64")

    def fit_transform(self, X: Any) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, Y: Any) -> list[str]:
        self._require_fitted()
        return [self.classes_[int(i)] if 0 <= int(i) < len(self.classes_) else "<unknown>"
                for i in np.asarray(Y).ravel()]

    @property
    def num_classes(self) -> int:
        return len(self.classes_) if self.fitted else 0

    def to_dict(self) -> dict[str, Any]:
        return {"name": "label", "classes": self.classes_}

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "LabelEncoder":
        encoder = cls()
        encoder.classes_ = list(state.get("classes", []))
        encoder.fitted = bool(encoder.classes_)
        return encoder

    def _require_fitted(self) -> None:
        if not self.fitted:
            raise PreprocessingError("LabelEncoder is not fitted")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _ordered_unique(values: Sequence[Any]) -> list[str]:
    seen: list[str] = []
    for value in values:
        key = str(value)
        if key not in seen and key != "nan":
            seen.append(key)
    return seen
