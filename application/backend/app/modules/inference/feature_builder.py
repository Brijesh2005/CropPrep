"""Feature Builder -> ModelInput (pipeline steps 3-4).

Turns a :class:`ResolvedLocation` into the exact tensors the exported
CropFusion model expects: a tabular vector of scaled numeric features plus
ordinal categorical codes, and a constant-filled NDVI/EVI patch sequence with
a matching temporal mask (the release package does not ship GeoTIFFs, so the
per-location raster means from ``village_metadata.parquet`` stand in for the
satellite patch; any patch value is trace-time constant, so this is
deliberately *not* a real satellite read — it keeps the exported model's
temporal/image branch active with a neutral input).

The feature contract is read from ``configs/inference.yaml`` (the
inference-time contract the release build writes)::

    feature_order:        [Area, Rainfall, Temperature, Humidity, price]
    categorical_features: [soil_type, irrigation]
    input_dim:            7          (5 scaled numeric + 2 ordinal codes)
    image_size:           224
    temporal_observations: 1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from app.core.exceptions import InferenceError
from app.modules.inference.location_resolver import ResolvedLocation
from inference_package.release import ReleasePackage


@dataclass(frozen=True, slots=True)
class ModelInput:
    tabular: torch.Tensor  # shape (1, input_dim) — scaled numeric + ordinal codes
    ndvi: torch.Tensor  # shape (1, T, 1, image_size, image_size)
    evi: torch.Tensor  # shape (1, T, 1, image_size, image_size)
    temporal_mask: torch.Tensor  # shape (1, T)
    feature_names: tuple[str, ...]
    raw_values: dict[str, float]


class FeatureBuilder:
    """Assembles and scales the feature tensors for one prediction."""

    def __init__(self, package: ReleasePackage) -> None:
        self._package = package
        inference_config = package.inference_config or {}

        self._feature_order: tuple[str, ...] = self._names(
            inference_config.get("feature_order") or package.model_config.get("feature_order")
        )
        if not self._feature_order:
            raise InferenceError("configs/inference.yaml is missing 'feature_order'")

        self._categorical_features: tuple[str, ...] = self._names(
            inference_config.get("categorical_features")
        )
        self._categorical_cardinalities: tuple[int, ...] = self._cardinalities(
            package.model_config
        )
        self._image_size = int(
            inference_config.get("image_size")
            or (package.model_config.get("image_encoder") or {}).get("input_size")
            or 224
        )
        self._temporal = max(
            1, int(inference_config.get("temporal_observations") or 1)
        )
        self._scaler = package.scaler
        self._device = package.device

    # ------------------------------------------------------------------ #
    # Build
    # ------------------------------------------------------------------ #

    def build(self, location: ResolvedLocation) -> ModelInput:
        merged: dict[str, Any] = {
            **location.village_metadata,
            **location.historical_context,
            "lon": location.lon,
            "lat": location.lat,
        }

        raw_values: dict[str, float] = {}
        missing: list[str] = []

        numeric = []
        for name in self._feature_order:
            value = merged.get(name)
            if value is None:
                missing.append(name)
                value = 0.0
            raw_values[name] = float(value)
            numeric.append(float(value))

        categorical = []
        for i, name in enumerate(self._categorical_features):
            value = merged.get(name)
            if value is None:
                missing.append(name)
                value = 0
            raw_values[name] = float(value)
            code = int(value)
            cardinality = self._cardinality(i)
            if cardinality is not None and not (0 <= code < cardinality):
                # Never let an out-of-range code blow past the embedding table.
                code = 0 if cardinality <= 0 else cardinality - 1
            categorical.append(float(code))

        if missing:
            from app.core.logging import get_logger

            get_logger("feature-builder").warning(
                "missing features filled with defaults",
                village=location.village,
                missing=missing,
            )

        vector = np.array([numeric], dtype="float64")
        if self._scaler is not None and hasattr(self._scaler, "transform"):
            vector = self._scaler.transform(vector)
        vector = np.concatenate([vector, np.asarray([categorical], dtype="float64")], axis=1)

        ndvi_value = self._patch_value(merged.get("NDVI"))
        evi_value = self._patch_value(merged.get("EVI"))

        tabular = torch.tensor(vector, dtype=torch.float32, device=self._device)
        ndvi = torch.full(
            (1, self._temporal, 1, self._image_size, self._image_size),
            ndvi_value,
            dtype=torch.float32,
            device=self._device,
        )
        evi = torch.full(
            (1, self._temporal, 1, self._image_size, self._image_size),
            evi_value,
            dtype=torch.float32,
            device=self._device,
        )
        temporal_mask = torch.ones(
            (1, self._temporal), dtype=torch.float32, device=self._device
        )

        return ModelInput(
            tabular=tabular,
            ndvi=ndvi,
            evi=evi,
            temporal_mask=temporal_mask,
            feature_names=self._feature_order + self._categorical_features,
            raw_values=raw_values,
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _names(value: Any) -> tuple[str, ...]:
        if not value:
            return ()
        return tuple(str(name) for name in value)

    @staticmethod
    def _cardinalities(model_config: dict[str, Any]) -> tuple[int, ...]:
        raw = (model_config.get("tabular") or {}).get("categorical_cardinalities")
        if not raw:
            return ()
        return tuple(int(c) for c in raw)

    def _cardinality(self, index: int) -> int | None:
        if index < len(self._categorical_cardinalities):
            return self._categorical_cardinalities[index]
        return None

    @staticmethod
    def _patch_value(value: Any) -> float:
        """Map a mean vegetation index (roughly -1..1) to a [0, 1] patch fill."""
        if value is None:
            return 0.5
        try:
            v = float(value)
        except (TypeError, ValueError):
            return 0.5
        if np.isnan(v):
            return 0.5
        return float(np.clip(0.5 + 0.5 * v, 0.0, 1.0))


__all__ = ["FeatureBuilder", "ModelInput"]
