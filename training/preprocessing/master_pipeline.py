"""Master preprocessing pipeline: AgriculturalObservation -> AI-ready sample.

The :class:`Preprocessor` orchestrates the per-modality pipelines and turns an
:class:`~training.stam.observation.AgriculturalObservation` into
a tensors dict directly consumable by a PyTorch Dataset/DataLoader::

    {
      "observation_id": str,
      "tabular":       torch.Tensor [F],
      "ndvi":          torch.Tensor [T, 1, H, W],
      "evi":           torch.Tensor [T, 1, H, W],
      "temporal_mask": torch.Tensor [T],
      "crop_label":    torch.Tensor scalar (int64),
      "yield_label":   torch.Tensor scalar (float32),
      "metadata":      dict,
    }

Fit happens **on training observations only** (no leakage). Transform is
applied lazily per sample so the dataset stays memory efficient.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from training.stam.exceptions import PatchOutOfBoundsError
from training.stam.observation import AgriculturalObservation

from .augmentations import ImageAugmentation
from .config import PreprocessingConfig, load_preprocessing_config
from .exceptions import FitError, PreprocessingError, SampleRejectedError
from .image_pipeline import ImagePipeline
from .label_pipeline import LabelPipeline
from .logger import get_logger
from .tabular_pipeline import TabularPipeline
from .temporal_pipeline import TemporalPipeline
from .validators import FilterDecision, filter_observations, filter_observation

logger = get_logger("master")

__all__ = ["Preprocessor"]


class Preprocessor:
    """Master preprocessing facade.

    Args:
        config: Validated :class:`PreprocessingConfig`.
        tabular / image / temporal / label: Optional per-modality pipeline
            overrides (injected for tests).
        augmentation: Optional :class:`ImageAugmentation` override.
    """

    def __init__(
        self,
        config: PreprocessingConfig | None = None,
        *,
        tabular: TabularPipeline | None = None,
        image: ImagePipeline | None = None,
        temporal: TemporalPipeline | None = None,
        label: LabelPipeline | None = None,
        augmentation: ImageAugmentation | None = None,
    ) -> None:
        self.config = config or PreprocessingConfig()
        self.tabular = tabular or TabularPipeline(self.config.tabular)
        self.image = image or ImagePipeline(self.config.image)
        self.temporal = temporal or TemporalPipeline(self.config.temporal)
        self.label = label or LabelPipeline(self.config.label)
        self.augmentation = augmentation or ImageAugmentation(self.config.augmentation)
        self.fitted = False
        self.filter_stats: dict[str, int] = {}

    @classmethod
    def from_config(cls, config_path: str | Path | None = None) -> "Preprocessor":
        """Build a Preprocessor from a YAML config file / ``PRE_*`` env vars."""
        return cls(load_preprocessing_config(config_path))

    # ------------------------------------------------------------------ #
    # Quality filtering
    # ------------------------------------------------------------------ #

    def filter(self, observations: Sequence[Any]) -> tuple[list[Any], list[FilterDecision]]:
        """Reject observations that fail the quality thresholds."""
        accepted, decisions = filter_observations(list(observations), self.config.quality)
        self.filter_stats = {"accepted": len(accepted), "total": len(observations)}
        return accepted, decisions

    def filter_one(self, observation: Any) -> FilterDecision:
        return filter_observation(observation, self.config.quality)

    # ------------------------------------------------------------------ #
    # Fit / transform
    # ------------------------------------------------------------------ #

    def fit(
        self,
        train_observations: Sequence[Any],
        *,
        extractor: Any | None = None,
    ) -> "Preprocessor":
        """Fit every sub-pipeline on the training observations only.

        Args:
            train_observations: Accepted training observations.
            extractor: ``callable(path, lon, lat, size=...) -> RasterPatch``
                used to read patches (e.g. ``STAM.get_patch``).
        """
        self.tabular.fit(train_observations)
        self.image.fit(train_observations, extractor=extractor)
        self.temporal.fit(train_observations)
        self.label.fit(train_observations)
        self.fitted = True
        logger.info(
            "Preprocessor fitted",
            extra={
                "samples": len(train_observations),
                "features": len(self.tabular.feature_names),
                "num_classes": self.label.num_classes,
            },
        )
        return self

    def transform(
        self,
        observation: Any,
        *,
        extractor: Any | None = None,
        augment: bool = False,
    ) -> dict[str, Any]:
        """Transform one observation into an AI-ready sample dict."""
        self._require_fitted()

        # Quality re-check (defensive).
        decision = filter_observation(observation, self.config.quality)
        if not decision.accepted:
            raise SampleRejectedError(
                f"Observation {decision.observation_id} rejected: "
                + ";".join(decision.reasons),
                detail=decision.reasons,
            )

        # -- Tabular ------------------------------------------------------ #
        tabular_tensor = self.tabular.transform(observation)

        # -- Images (per-pair patches) ------------------------------------ #
        ndvi_tensors, evi_tensors, dates = self._extract_sequences(
            observation, extractor
        )

        # -- Temporal ------------------------------------------------------ #
        ndvi_seq, evi_seq, temporal_mask = self.temporal.transform_sequence(
            ndvi_tensors, evi_tensors, dates
        )

        # -- Augmentation (train only) ------------------------------------ #
        if augment:
            ndvi_seq = self.augmentation(ndvi_seq)
            evi_seq = self.augmentation(evi_seq)

        # -- Labels -------------------------------------------------------- #
        crop_label, yield_label = self.label.transform(observation)

        return {
            "observation_id": str(observation.observation_id),
            "tabular": tabular_tensor,
            "ndvi": ndvi_seq,
            "evi": evi_seq,
            "temporal_mask": temporal_mask,
            "crop_label": crop_label,
            "yield_label": yield_label,
            "metadata": _sample_metadata(observation),
        }

    def fit_transform(
        self,
        observations: Sequence[Any],
        *,
        extractor: Any | None = None,
    ) -> list[dict[str, Any]]:
        """Fit on ``observations`` then transform all of them (train pass)."""
        self.fit(observations, extractor=extractor)
        return [
            self.transform(obs, extractor=extractor, augment=False)
            for obs in observations
        ]

    # ------------------------------------------------------------------ #
    # Validation / summary / persistence
    # ------------------------------------------------------------------ #

    def validate(self, observation: Any) -> list[Any]:
        """Run every sub-pipeline's checks; empty list means valid."""
        issues: list[Any] = []
        issues.extend(self.tabular.validate(observation))
        issues.extend(self.image.validate(observation))
        issues.extend(self.temporal.validate(observation))
        issues.extend(self.label.validate(observation))
        return issues

    def summary(self) -> dict[str, Any]:
        return {
            "fitted": self.fitted,
            "config": self.config.model_dump_json(),
            "tabular": self.tabular.summary(),
            "image": self.image.summary(),
            "temporal": self.temporal.summary(),
            "label": self.label.summary(),
            "augmentation_enabled": self.augmentation.enabled,
            "filter_stats": self.filter_stats,
        }

    def save(self, directory: str | Path) -> Path:
        """Persist fitted artifacts under ``directory``."""
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        self.tabular.save(out)
        self.image.save(out)
        self.temporal.save(out)
        self.label.save(out)
        import json

        (out / "preprocessor_meta.json").write_text(
            json.dumps({"fitted": self.fitted, "filter_stats": self.filter_stats},
                       indent=2),
            encoding="utf-8",
        )
        return out

    @classmethod
    def load(cls, directory: str | Path) -> "Preprocessor":
        """Restore a fitted Preprocessor from ``directory``."""
        import json

        base = Path(directory)
        preprocessor = cls()
        preprocessor.tabular = TabularPipeline.load(base)
        preprocessor.image = ImagePipeline.load(base)
        preprocessor.temporal = TemporalPipeline.load(base)
        preprocessor.label = LabelPipeline.load(base)
        meta = json.loads((base / "preprocessor_meta.json").read_text(encoding="utf-8"))
        preprocessor.fitted = bool(meta.get("fitted"))
        preprocessor.filter_stats = meta.get("filter_stats", {})
        return preprocessor

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _require_fitted(self) -> None:
        if not self.fitted:
            raise FitError("Preprocessor has not been fitted; call fit() first")

    def _extract_sequences(
        self, observation: Any, extractor: Any | None
    ) -> tuple[list[Any], list[Any], list[Any]]:
        ndvi_tensors: list[Any] = []
        evi_tensors: list[Any] = []
        dates: list[Any] = []
        pairs = observation.sequence.pairs
        if not pairs:
            return [], [], []

        if extractor is None:
            raise PreprocessingError(
                "A patch extractor is required to build image tensors "
                "(pass extractor, e.g. STAM.get_patch)"
            )

        lon, lat = observation.location.lon, observation.location.lat
        size = observation.patch_size or self.config.image.size
        for pair in pairs:
            ndvi_tensor = None
            if pair.ndvi is not None:
                patch = self._extract_patch(
                    extractor, pair.ndvi.path, lon, lat, size, observation
                )
                if patch is not None:
                    ndvi_tensor = self.image.transform_patch(
                        patch.array, "NDVI", mask=getattr(patch, "mask", None)
                    )
            evi_tensor = None
            if pair.evi is not None:
                patch = self._extract_patch(
                    extractor, pair.evi.path, lon, lat, size, observation
                )
                if patch is not None:
                    evi_tensor = self.image.transform_patch(
                        patch.array, "EVI", mask=getattr(patch, "mask", None)
                    )
            ndvi_tensors.append(ndvi_tensor)
            evi_tensors.append(evi_tensor)
            dates.append(pair.date)
        return ndvi_tensors, evi_tensors, dates

    def _extract_patch(
        self,
        extractor: Any,
        path: str,
        lon: float,
        lat: float,
        size: int,
        observation: Any,
    ) -> Any | None:
        """Extract a patch; return None when the point is outside the raster.

        Defensive net behind the sequence-level spatial filter: if the point
        still misses a raster (projection/metadata drift), the band is treated
        as missing instead of aborting the whole sample.
        """
        try:
            return extractor(path, lon, lat, size=size)
        except PatchOutOfBoundsError:
            logger.warning(
                "Patch out of bounds; band treated as missing",
                extra={
                    "path": path,
                    "lon": lon, "lat": lat,
                    "observation_id": observation.id,
                },
            )
            return None


def _sample_metadata(observation: AgriculturalObservation) -> dict[str, Any]:
    return {
        "lon": observation.location.lon,
        "lat": observation.location.lat,
        "year": observation.temporal.year,
        "season": observation.temporal.season,
        "quality_score": observation.quality.overall_score,
        "patch_size": observation.patch_size,
        "crop": observation.crop,
        "yield_value": observation.yield_value,
    }
