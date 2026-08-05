"""Image (NDVI/EVI patch) pipeline: raw patch -> normalized tensor.

Handles NaN/invalid pixels, physical-range clipping, normalization
(minmax / standard / identity), resizing and tensor conversion. Spatial
information is preserved — no handcrafted vegetation descriptors are
computed, per the SDD.

Normalization modes:

* ``minmax`` — map the physical index range (e.g. NDVI [-1, 1]) to [0, 1].
  No patch reads are needed for fitting.
* ``standard`` — fit a global mean/std from a bounded sample of patches.
* ``identity`` — pass values through (after NaN/invalid handling + clip).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import ImageConfig
from .exceptions import FitError, PreprocessingError
from .interfaces import Pipeline
from .logger import get_logger
from .utils import to_float_tensor

logger = get_logger("image")

#: Cap on the number of patches sampled to fit standard normalization.
_FIT_SAMPLE_CAP = 50


class ImagePipeline(Pipeline):
    """Per-modality pipeline for NDVI/EVI patch tensors.

    Args:
        config: Image processing settings.
    """

    def __init__(self, config: ImageConfig | None = None) -> None:
        self.config = config or ImageConfig()
        self.fitted = False
        self.ndvi_mean: float = 0.0
        self.ndvi_std: float = 1.0
        self.evi_mean: float = 0.0
        self.evi_std: float = 1.0
        self._patch_samples: int = 0

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #

    def fit(
        self,
        samples: Sequence[Any],
        *,
        extractor: Any | None = None,
    ) -> "ImagePipeline":
        """Fit normalization statistics.

        For ``minmax`` the physical ranges from config are used (fast). For
        ``standard`` a bounded sample of patches is read through the
        ``extractor`` to estimate the global mean/std per index.

        Args:
            samples: Training observations.
            extractor: ``callable(path, lon, lat, size=...) -> RasterPatch``.
        """
        if self.config.normalize == "standard":
            if extractor is None:
                raise PreprocessingError(
                    "standard image normalization requires a patch extractor during fit"
                )
            ndvi_values: list[float] = []
            evi_values: list[float] = []
            sampled = 0
            for observation in samples:
                if sampled >= _FIT_SAMPLE_CAP:
                    break
                for pair in observation.sequence.pairs:
                    if pair.ndvi is not None and pair.evi is not None:
                        patch_n = _extract(extractor, pair.ndvi.path, observation)
                        patch_e = _extract(extractor, pair.evi.path, observation)
                        ndvi_values.extend(_finite_values(patch_n.array))
                        evi_values.extend(_finite_values(patch_e.array))
                        sampled += 1
                        break
            if ndvi_values:
                self.ndvi_mean = float(np.mean(ndvi_values))
                self.ndvi_std = float(np.std(ndvi_values)) or 1.0
            if evi_values:
                self.evi_mean = float(np.mean(evi_values))
                self.evi_std = float(np.std(evi_values)) or 1.0
            self._patch_samples = sampled
        self.fitted = True
        logger.info(
            "Image pipeline fitted",
            extra={"normalize": self.config.normalize, "size": self.config.size,
                   "sampled": self._patch_samples},
        )
        return self

    # ------------------------------------------------------------------ #
    # Transform
    # ------------------------------------------------------------------ #

    def transform_patch(
        self,
        array: np.ndarray,
        index_type: str,
        *,
        mask: np.ndarray | None = None,
    ) -> Any:
        """Normalize one patch array into a ``[1, size, size]`` float tensor."""
        self._require_fitted()
        patch = np.asarray(array, dtype="float32")
        if patch.ndim == 3:
            patch = patch[0]
        if patch.ndim != 2:
            raise PreprocessingError(
                f"Patch must be 2-D, got shape {patch.shape}"
            )

        # NaN / invalid handling.
        patch = self._handle_nan(patch, index_type)
        if mask is not None:
            patch = self._handle_invalid(patch, mask)

        # Physical-range clipping.
        lo, hi = self._physical_range(index_type)
        if self.config.clip:
            patch = np.clip(patch, lo, hi)

        # Normalization.
        if self.config.normalize == "minmax":
            span = (hi - lo) or 1.0
            patch = (patch - lo) / span
        elif self.config.normalize == "standard":
            mean = self.ndvi_mean if index_type == "NDVI" else self.evi_mean
            std = self.ndvi_std if index_type == "NDVI" else self.evi_std
            patch = (patch - mean) / (std or 1.0)
        # identity: unchanged

        # Resize / pad to the configured size.
        patch = self._to_size(patch)

        # Channel dimension.
        patch = patch[np.newaxis, :, :]
        return to_float_tensor(patch)

    def transform(self, observation: Any) -> Any:
        raise NotImplementedError(
            "ImagePipeline operates per-patch; use transform_patch() from the "
            "master pipeline / dataset"
        )

    def validate(self, observation: Any) -> list[Any]:
        issues: list[str] = []
        if observation.patch_size and observation.patch_size != self.config.size:
            issues.append(
                f"patch_size_mismatch(stam={observation.patch_size}, "
                f"target={self.config.size})"
            )
        return issues

    def summary(self) -> dict[str, Any]:
        return {
            "fitted": self.fitted,
            "size": self.config.size,
            "normalize": self.config.normalize,
            "ndvi_range": list(self.config.ndvi_range),
            "evi_range": list(self.config.evi_range),
            "ndvi_mean": self.ndvi_mean,
            "ndvi_std": self.ndvi_std,
            "evi_mean": self.evi_mean,
            "evi_std": self.evi_std,
            "sampled_patches": self._patch_samples,
        }

    def save(self, directory: str | Path) -> Path:
        import json

        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        state = {
            "size": self.config.size,
            "normalize": self.config.normalize,
            "ndvi_mean": self.ndvi_mean,
            "ndvi_std": self.ndvi_std,
            "evi_mean": self.evi_mean,
            "evi_std": self.evi_std,
            "sampled_patches": self._patch_samples,
        }
        (out / "image_pipeline.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
        return out

    @classmethod
    def load(cls, directory: str | Path) -> "ImagePipeline":
        import json

        state = json.loads((Path(directory) / "image_pipeline.json").read_text(encoding="utf-8"))
        pipeline = cls(ImageConfig(size=state["size"], normalize=state["normalize"]))
        pipeline.ndvi_mean = state["ndvi_mean"]
        pipeline.ndvi_std = state["ndvi_std"]
        pipeline.evi_mean = state["evi_mean"]
        pipeline.evi_std = state["evi_std"]
        pipeline._patch_samples = state["sampled_patches"]
        pipeline.fitted = True
        return pipeline

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _require_fitted(self) -> None:
        if not self.fitted:
            raise FitError("ImagePipeline has not been fitted")

    def _physical_range(self, index_type: str) -> tuple[float, float]:
        return self.config.ndvi_range if index_type == "NDVI" else self.config.evi_range

    def _handle_nan(self, patch: np.ndarray, index_type: str) -> np.ndarray:
        if not np.isfinite(patch).all():
            if self.config.nan_policy == "mean":
                finite = patch[np.isfinite(patch)]
                fill = float(finite.mean()) if finite.size else 0.0
            else:  # zero / drop
                fill = 0.0
            patch = np.where(np.isfinite(patch), patch, fill)
        return patch

    def _handle_invalid(self, patch: np.ndarray, mask: np.ndarray) -> np.ndarray:
        invalid = ~np.asarray(mask, dtype=bool)
        if invalid.any():
            patch = np.where(invalid, 0.0, patch)
        return patch

    def _to_size(self, patch: np.ndarray) -> np.ndarray:
        height, width = patch.shape
        target = self.config.size
        if (height, width) == (target, target):
            return patch
        if self.config.resize:
            return _resize(patch, target)
        if self.config.pad:
            return _pad_to(patch, target)
        raise PreprocessingError(
            f"Patch {patch.shape} does not match target {target} and "
            "resize/pad are disabled"
        )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _extract(extractor: Any, path: str, observation: Any) -> Any:
    """Extract a patch for an image path at the observation's location."""
    return extractor(
        path,
        observation.location.lon,
        observation.location.lat,
        size=observation.patch_size or None,
    )


def _finite_values(array: np.ndarray) -> list[float]:
    flat = np.asarray(array, dtype="float32").ravel()
    return flat[np.isfinite(flat)].tolist()


def _resize(patch: np.ndarray, target: int) -> np.ndarray:
    """Bilinear resize a 2-D patch to ``target x target`` (torch)."""
    import torch

    tensor = torch.from_numpy(patch.astype("float32")).unsqueeze(0).unsqueeze(0)
    resized = torch.nn.functional.interpolate(
        tensor, size=(target, target), mode="bilinear", align_corners=False
    )
    return resized[0, 0].numpy()


def _pad_to(patch: np.ndarray, target: int) -> np.ndarray:
    height, width = patch.shape
    out = np.zeros((target, target), dtype="float32")
    out[: min(height, target), : min(width, target)] = patch[
        : min(height, target), : min(width, target)
    ]
    return out
