"""Label pipeline: crop + yield labels -> tensors.

Crop labels are encoded with a persisted :class:`LabelEncoder` (or one-hot),
yield targets are scaled (standard / minmax / none) for regression. The label
encoder is saved alongside the pipeline so inference can decode predictions
back to crop names.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import LabelConfig
from .exceptions import FitError, PreprocessingError
from .interfaces import Pipeline, Transformer
from .logger import get_logger
from .transforms import LabelEncoder, MinMaxScaler, StandardScaler
from .utils import to_float_tensor, to_long_tensor

logger = get_logger("label")


class LabelPipeline(Pipeline):
    """Per-modality pipeline for crop/yield training labels.

    Args:
        config: Label processing settings.
    """

    def __init__(self, config: LabelConfig | None = None) -> None:
        self.config = config or LabelConfig()
        self.fitted = False
        self.crop_encoder: LabelEncoder | None = None
        self.yield_scaler: Transformer | None = None
        # R5.2 Task 5/7: scale-consistency diagnostics captured at fit time so
        # a mixed-unit yield target (e.g. kg/ha vs a normalized proxy) is
        # surfaced instead of silently distorting the regression target.
        self.yield_scale_stats: dict[str, Any] | None = None
        self.warnings: list[str] = []

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #

    def fit(self, samples: Sequence[Any]) -> "LabelPipeline":
        """Fit the crop label encoder and yield scaler on training samples."""
        crops = [s.crop for s in samples if s.crop is not None]
        if crops:
            self.crop_encoder = LabelEncoder().fit(crops)

        yields = [s.yield_value for s in samples if s.yield_value is not None]
        if yields:
            matrix = np.asarray(yields, dtype="float64").reshape(-1, 1)
            scaler_name = self.config.yield_scaler
            if scaler_name == "standard":
                self.yield_scaler = StandardScaler().fit(matrix)
            elif scaler_name == "minmax":
                self.yield_scaler = MinMaxScaler().fit(matrix)
            else:
                self.yield_scaler = None
            self._diagnose_yield_scale(yields)

        self.fitted = True
        logger.info(
            "Label pipeline fitted",
            extra={"num_classes": self.crop_encoder.num_classes if self.crop_encoder else 0},
        )
        return self

    def _diagnose_yield_scale(self, yields: list[float]) -> None:
        """Record scale-consistency diagnostics for the yield regression target.

        R5.2 Task 5/7: flag a target whose raw values span several orders of
        magnitude (classic mixed-units signature, e.g. kg/ha village yields
        mixed with a normalized per-district proxy) or whose scaled values
        collapse to a handful of distinct points (the model cannot regress
        anything beyond a constant for the bulk of the corpus).
        """
        values = np.asarray(yields, dtype="float64")
        stats: dict[str, Any] = {
            "n": int(len(values)),
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "dynamic_range_ratio": float(values.max() / max(values.min(), 1e-9)),
        }
        if stats["dynamic_range_ratio"] > 1e3:
            self.warnings.append(
                f"yield target raw dynamic range {stats['dynamic_range_ratio']:.1e} "
                f"(min={stats['min']:.4g}, max={stats['max']:.4g}) — mixed units "
                "(e.g. kg/ha vs a normalized proxy) distort the regression target"
            )
        if self.yield_scaler is not None:
            scaled = self.yield_scaler.transform(values.reshape(-1, 1))[:, 0]
            distinct = int(len(set(round(float(v), 4) for v in scaled)))
            stats["scaled_distinct_values"] = distinct
            stats["scaled_distinct_ratio"] = round(distinct / max(len(values), 1), 4)
            if distinct <= 1:
                self.warnings.append(
                    "yield target collapses to a SINGLE scaled value for the "
                    "entire corpus — the regression target carries no signal"
                )
            elif distinct <= max(2, len(values) // 10):
                self.warnings.append(
                    f"yield target collapses to only {distinct} distinct scaled "
                    f"values over {len(values)} samples — the bulk regresses to "
                    "a constant"
                )
        self.yield_scale_stats = stats
        for warning in self.warnings:
            logger.warning(warning)

    # ------------------------------------------------------------------ #
    # Transform
    # ------------------------------------------------------------------ #

    def transform(self, observation: Any) -> tuple[Any, Any]:
        """Return ``(crop_tensor, yield_tensor)`` for an observation.

        * crop_tensor — int64 scalar (label encoding) or ``[C]`` one-hot.
        * yield_tensor — float32 scalar (scaled).
        """
        self._require_fitted()

        # -- Crop --------------------------------------------------------- #
        if self.config.crop_encoding == "onehot" and self.crop_encoder is not None:
            code = self.crop_encoder.transform([observation.crop])[0]
            vector = np.zeros(self.crop_encoder.num_classes, dtype="float32")
            if code >= 0:
                vector[int(code)] = 1.0
            crop_tensor = to_float_tensor(vector)
        else:
            code = self.crop_encoder.transform([observation.crop])[0] if self.crop_encoder else -1
            crop_tensor = to_long_tensor(np.asarray([code], dtype="int64"))[0]

        # -- Yield -------------------------------------------------------- #
        if observation.yield_value is None:
            yield_value = 0.0
        else:
            yield_value = float(observation.yield_value)
        if self.yield_scaler is not None:
            scaled = self.yield_scaler.transform(np.asarray([[yield_value]], dtype="float64"))
            yield_value = float(scaled[0, 0])
        yield_tensor = to_float_tensor(np.asarray([yield_value], dtype="float32"))[0]

        return crop_tensor, yield_tensor

    def inverse_crop(self, codes: Any) -> list[str]:
        """Decode integer codes back to crop names (for evaluation)."""
        self._require_fitted()
        if self.crop_encoder is None:
            raise PreprocessingError("No crop encoder fitted")
        return self.crop_encoder.inverse_transform(codes)

    def validate(self, observation: Any) -> list[Any]:
        issues: list[str] = []
        if observation.crop is None:
            issues.append("missing_crop_label")
        if observation.yield_value is None:
            issues.append("missing_yield_label")
        return issues

    @property
    def num_classes(self) -> int:
        return self.crop_encoder.num_classes if self.crop_encoder else 0

    def summary(self) -> dict[str, Any]:
        return {
            "fitted": self.fitted,
            "crop_encoding": self.config.crop_encoding,
            "num_classes": self.num_classes,
            "classes": self.crop_encoder.classes_ if self.crop_encoder else [],
            "yield_scaler": self.config.yield_scaler,
            "yield_scale_stats": self.yield_scale_stats,
            "warnings": list(self.warnings),
        }

    def save(self, directory: str | Path) -> Path:
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        state = {
            "crop_encoder": self.crop_encoder.to_dict() if self.crop_encoder else None,
            "yield_scaler": self.yield_scaler.to_dict() if self.yield_scaler else None,
        }
        (out / "label_pipeline.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
        return out

    @classmethod
    def load(cls, directory: str | Path) -> "LabelPipeline":
        state = json.loads((Path(directory) / "label_pipeline.json").read_text(encoding="utf-8"))
        pipeline = cls()
        if state.get("crop_encoder"):
            pipeline.crop_encoder = LabelEncoder.from_dict(state["crop_encoder"])
        if state.get("yield_scaler"):
            name = state["yield_scaler"]["name"]
            pipeline.yield_scaler = (
                StandardScaler.from_dict(state["yield_scaler"])
                if name == "standard"
                else MinMaxScaler.from_dict(state["yield_scaler"])
            )
        pipeline.fitted = True
        return pipeline

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _require_fitted(self) -> None:
        if not self.fitted:
            raise FitError("LabelPipeline has not been fitted")
