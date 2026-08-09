"""Tabular feature pipeline: fields dict -> scaled/encoded feature tensor.

Handles missing values, outlier clipping, feature scaling, categorical
encoding, constant-feature removal, correlation analysis and per-feature
statistics. All fitted parameters (scalers, encoders, fill values, clip
bounds) are persisted so train/val/test transforms are consistent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import TabularConfig
from .exceptions import FitError, PreprocessingError
from .interfaces import Pipeline, Transformer
from .logger import get_logger
from .transforms import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)
from .utils import is_numeric, to_float_tensor

logger = get_logger("tabular")

_MISSING_CATEGORY = "__missing__"


class TabularPipeline(Pipeline):
    """Per-modality pipeline for tabular agricultural features.

    Args:
        config: Tabular processing settings.
    """

    def __init__(self, config: TabularConfig | None = None) -> None:
        self.config = config or TabularConfig()
        self.fitted = False
        self.numeric_features: list[str] = []
        self.categorical_features: list[str] = []
        self.feature_names: list[str] = []
        self.scaler: Transformer | None = None
        self.encoder: Transformer | None = None
        self.missing_fill: dict[str, float] = {}
        self.clip_bounds: dict[str, tuple[float, float]] = {}
        self.dropped_constant: list[str] = []
        self.dropped_correlated: list[str] = []
        #: Configured feature columns absent from the training data.
        self.missing_columns: list[str] = []
        self._categorical_columns: list[str] = []

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #

    def fit(self, samples: Sequence[Any]) -> "TabularPipeline":
        """Fit scalers/encoders and resolve feature columns on training samples."""
        fields = [dict(s.tabular.fields) for s in samples]
        keys = _all_keys(fields)
        excluded = {str(k).lower() for k in self.config.exclude_columns}

        if self.config.numeric_features or self.config.categorical_features:
            numeric = list(self.config.numeric_features)
            categorical = list(self.config.categorical_features)
        else:
            numeric, categorical = _infer_types(fields, keys, excluded)

        # Remove constant features (single unique value across samples).
        numeric = self._drop_constants(fields, numeric)

        # Correlation analysis on numeric columns.
        if self.config.max_correlation is not None:
            numeric = self._drop_correlated(fields, numeric)

        self.numeric_features = numeric
        self.categorical_features = categorical
        self._categorical_columns = categorical

        # -- Missing fill values ------------------------------------------ #
        self.missing_fill = self._compute_fill_values(fields, numeric)

        # -- Outlier clip bounds ------------------------------------------ #
        self.clip_bounds = self._compute_clip_bounds(fields, numeric)

        # -- Numeric scaling ---------------------------------------------- #
        if numeric:
            matrix = _numeric_matrix(fields, numeric, self.missing_fill, self.clip_bounds)
            self.scaler = _make_scaler(self.config.scaler)
            if self.scaler is not None:
                self.scaler.fit(matrix)
            self.numeric_feature_stats = _column_stats(matrix, numeric)
        else:
            self.numeric_feature_stats = {}

        # -- Categorical encoding ----------------------------------------- #
        if categorical:
            cat_matrix = _categorical_matrix(fields, categorical)
            self.encoder = _make_encoder(self.config.categorical_encoding)
            if self.encoder is not None:
                # Map column name -> position so one-hot feature names expand
                # per-column categories (not always categories_[0]).
                self.encoder._col_index = {col: i for i, col in enumerate(categorical)}
                self.encoder.fit(cat_matrix)

        self.feature_names = list(numeric)
        if self.config.categorical_encoding == "onehot" and self.encoder is not None:
            for col in categorical:
                categories = _encoder_categories(self.encoder, col)
                self.feature_names.extend(f"{col}={cat}" for cat in categories)
        else:
            self.feature_names.extend(categorical)

        self.fitted = True
        logger.info(
            "Tabular pipeline fitted",
            extra={"numeric": len(numeric), "categorical": len(categorical),
                   "features": len(self.feature_names)},
        )
        return self

    # ------------------------------------------------------------------ #
    # Transform
    # ------------------------------------------------------------------ #

    def transform(self, observation: Any) -> Any:
        """Transform one observation's fields into a ``[F]`` float tensor."""
        self._require_fitted()
        fields = dict(observation.tabular.fields)

        vectors: list[np.ndarray] = []
        if self.numeric_features:
            matrix = _numeric_matrix(
                [fields], self.numeric_features, self.missing_fill, self.clip_bounds
            )
            scaled = self.scaler.transform(matrix) if self.scaler is not None else matrix
            vectors.append(np.asarray(scaled[0], dtype="float32"))

        if self.categorical_features:
            cat_matrix = _categorical_matrix([fields], self.categorical_features)
            if self.encoder is not None:
                encoded = self.encoder.transform(cat_matrix)[0]
            else:
                encoded = np.zeros(len(self.categorical_features), dtype="float32")
            vectors.append(np.asarray(encoded, dtype="float32"))

        vector = (
            np.concatenate(vectors) if vectors else np.zeros(0, dtype="float32")
        )
        return to_float_tensor(vector)

    # ------------------------------------------------------------------ #
    # Validation / summary / persistence
    # ------------------------------------------------------------------ #

    def validate(self, observation: Any) -> list[Any]:
        issues: list[str] = []
        fields = dict(observation.tabular.fields)
        for col in self.missing_columns:
            issues.append(f"missing_numeric_column:{col}")
        for col in self.numeric_features:
            if col not in fields:
                issues.append(f"missing_numeric_column:{col}")
        for col in self.categorical_features:
            if col not in fields:
                issues.append(f"missing_categorical_column:{col}")
        return issues

    def summary(self) -> dict[str, Any]:
        return {
            "fitted": self.fitted,
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "feature_count": len(self.feature_names),
            "feature_names": self.feature_names,
            "dropped_constant": self.dropped_constant,
            "dropped_correlated": self.dropped_correlated,
            "scaler": self.config.scaler,
            "encoding": self.config.categorical_encoding,
            "numeric_stats": getattr(self, "numeric_feature_stats", {}),
        }

    def save(self, directory: str | Path) -> Path:
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        state = {
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "feature_names": self.feature_names,
            "missing_fill": self.missing_fill,
            "clip_bounds": {k: list(v) for k, v in self.clip_bounds.items()},
            "dropped_constant": self.dropped_constant,
            "dropped_correlated": self.dropped_correlated,
            "missing_columns": self.missing_columns,
            "scaler": self.scaler.to_dict() if self.scaler else None,
            "encoder": self.encoder.to_dict() if self.encoder else None,
        }
        (out / "tabular_pipeline.json").write_text(
            json.dumps(state, indent=2, default=str), encoding="utf-8"
        )
        return out

    @classmethod
    def load(cls, directory: str | Path) -> "TabularPipeline":
        path = Path(directory) / "tabular_pipeline.json"
        state = json.loads(path.read_text(encoding="utf-8"))
        pipeline = cls()
        pipeline.numeric_features = list(state["numeric_features"])
        pipeline.categorical_features = list(state["categorical_features"])
        pipeline.feature_names = list(state["feature_names"])
        pipeline.missing_fill = state["missing_fill"]
        pipeline.clip_bounds = {k: tuple(v) for k, v in state["clip_bounds"].items()}
        pipeline.dropped_constant = list(state["dropped_constant"])
        pipeline.dropped_correlated = list(state["dropped_correlated"])
        pipeline.missing_columns = list(state.get("missing_columns", []))
        pipeline._categorical_columns = pipeline.categorical_features
        if state.get("scaler"):
            pipeline.scaler = _scaler_from_dict(state["scaler"])
        if state.get("encoder"):
            pipeline.encoder = _encoder_from_dict(state["encoder"])
        pipeline.fitted = True
        return pipeline

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _require_fitted(self) -> None:
        if not self.fitted:
            raise FitError("TabularPipeline has not been fitted")

    def _drop_constants(self, fields: list[dict], numeric: list[str]) -> list[str]:
        kept: list[str] = []
        for col in numeric:
            values = [_safe_float(f.get(col)) for f in fields]
            present = [v for v in values if v is not None]
            if not present:
                # Configured but absent from the data -> flagged, not dropped.
                self.missing_columns.append(col)
                continue
            if len(set(present)) <= 1:
                self.dropped_constant.append(col)
                continue
            kept.append(col)
        return kept

    def _drop_correlated(self, fields: list[dict], numeric: list[str]) -> list[str]:
        matrix = np.asarray(
            [[_safe_float(f.get(c)) or 0.0 for c in numeric] for f in fields],
            dtype="float64",
        )
        if matrix.shape[0] < 2 or matrix.shape[1] < 2:
            return numeric
        corr = np.corrcoef(matrix, rowvar=False)
        threshold = self.config.max_correlation or 0.95
        keep = np.ones(len(numeric), dtype=bool)
        for i in range(len(numeric)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(numeric)):
                if keep[j] and abs(corr[i, j]) > threshold:
                    keep[j] = False
        dropped = [numeric[i] for i in range(len(numeric)) if not keep[i]]
        self.dropped_correlated.extend(dropped)
        return [numeric[i] for i in range(len(numeric)) if keep[i]]

    def _compute_fill_values(self, fields: list[dict], numeric: list[str]) -> dict[str, float]:
        fills: dict[str, float] = {}
        for col in numeric:
            values = np.asarray([_safe_float(f.get(col)) for f in fields], dtype="float64")
            if self.config.handle_missing == "median":
                fills[col] = float(np.nanmedian(values)) if values.size else 0.0
            elif self.config.handle_missing == "mean":
                fills[col] = float(np.nanmean(values)) if values.size else 0.0
            else:  # zero / none
                fills[col] = 0.0
        return fills

    def _compute_clip_bounds(self, fields: list[dict], numeric: list[str]) -> dict[str, tuple[float, float]]:
        bounds: dict[str, tuple[float, float]] = {}
        method = self.config.outlier_method
        if method == "none":
            return bounds
        for col in numeric:
            values = np.asarray([_safe_float(f.get(col)) for f in fields], dtype="float64")
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            if method == "zscore":
                mean, std = float(np.mean(values)), float(np.std(values))
                k = self.config.outlier_threshold
                bounds[col] = (mean - k * std, mean + k * std)
            else:  # iqr
                q1, q3 = float(np.percentile(values, 25)), float(np.percentile(values, 75))
                iqr = q3 - q1
                bounds[col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        return bounds


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _all_keys(fields: list[dict]) -> list[str]:
    keys: list[str] = []
    for row in fields:
        for key in row:
            if key not in keys:
                keys.append(key)
    return keys


def _infer_types(fields: list[dict], keys: list[str], excluded: set[str]) -> tuple[list[str], list[str]]:
    numeric: list[str] = []
    categorical: list[str] = []
    for key in keys:
        if key.lower() in excluded:
            continue
        values = [f.get(key) for f in fields if key in f]
        if values and all(is_numeric(v) or v is None or v == "" for v in values):
            numeric.append(key)
        else:
            categorical.append(key)
    return numeric, categorical


def _numeric_matrix(
    fields: list[dict],
    columns: list[str],
    fills: dict[str, float],
    clip_bounds: dict[str, tuple[float, float]],
) -> np.ndarray:
    matrix = np.zeros((len(fields), len(columns)), dtype="float64")
    for row_index, row in enumerate(fields):
        for col_index, col in enumerate(columns):
            value = _safe_float(row.get(col))
            if value is None or not np.isfinite(value):
                value = fills.get(col, 0.0)
            bounds = clip_bounds.get(col)
            if bounds is not None:
                value = min(max(value, bounds[0]), bounds[1])
            matrix[row_index, col_index] = value
    return matrix


def _categorical_matrix(fields: list[dict], columns: list[str]) -> np.ndarray:
    matrix = np.empty((len(fields), len(columns)), dtype=object)
    for row_index, row in enumerate(fields):
        for col_index, col in enumerate(columns):
            value = row.get(col)
            matrix[row_index, col_index] = str(value) if value is not None and value != "" else _MISSING_CATEGORY
    return matrix


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _make_scaler(name: str) -> Transformer | None:
    factory = {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "robust": RobustScaler,
    }.get(name)
    return factory() if factory else None


def _make_encoder(name: str) -> Transformer | None:
    factory = {"onehot": OneHotEncoder, "ordinal": OrdinalEncoder}.get(name)
    return factory() if factory else None


def _encoder_categories(encoder: Transformer, column: str) -> list[str]:
    if isinstance(encoder, OneHotEncoder) and encoder.categories_:
        index = getattr(encoder, "_col_index", None)
        if index is not None:
            return encoder.categories_[index.get(column, 0)]
        return encoder.categories_[0]
    return []


def _column_stats(matrix: np.ndarray, columns: list[str]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for i, col in enumerate(columns):
        values = matrix[:, i]
        finite = values[np.isfinite(values)]
        if finite.size:
            stats[col] = {
                "min": float(finite.min()),
                "max": float(finite.max()),
                "mean": float(finite.mean()),
                "std": float(finite.std()),
            }
    return stats


def _scaler_from_dict(state: dict[str, Any]) -> Transformer:
    name = state.get("name")
    if name == "standard":
        return StandardScaler.from_dict(state)
    if name == "minmax":
        return MinMaxScaler.from_dict(state)
    if name == "robust":
        return RobustScaler.from_dict(state)
    raise PreprocessingError(f"Unknown scaler: {name}")


def _encoder_from_dict(state: dict[str, Any]) -> Transformer:
    name = state.get("name")
    if name == "onehot":
        return OneHotEncoder.from_dict(state)
    if name == "ordinal":
        return OrdinalEncoder.from_dict(state)
    raise PreprocessingError(f"Unknown encoder: {name}")
