"""Image / sequence feature builder.

:class:`ImageFeatureBuilder` summarises the observation's ordered NDVI/EVI
sequence. By default only *metadata* features are produced (record counts,
paired dates, resolution, temporal gaps, coverage) — no raster is ever read.

When :attr:`~training.feature_engineering.config.ImageFeatureConfig.
extract_patch_stats` is enabled an ``extractor`` callable
(``callable(path, lon, lat, size=...) -> RasterPatch`` — e.g.
``stam.get_patch``) is required and per-date NDVI/EVI patch statistics
(mean/std/min/max) are appended for the first ``max_dates`` paired dates.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Callable

from .config import ImageFeatureConfig
from .exceptions import MissingExtractorError
from .logger import get_logger

logger = get_logger("image")

_PATCH_STATS = ("mean", "std", "min", "max")


class ImageFeatureBuilder:
    """Summarise the image sequence of an observation into features."""

    def __init__(self, config: ImageFeatureConfig | None = None) -> None:
        self.config = config or ImageFeatureConfig()

    def build(
        self,
        observation: Any,
        *,
        prefix: str = "img",
        extractor: Callable[..., Any] | None = None,
    ) -> dict[str, Any]:
        """Build image-sequence features for an observation.

        Args:
            observation: An :class:`AgriculturalObservation`.
            prefix: Modality prefix (``""`` disables prefixing).
            extractor: Patch extractor required only when patch statistics
                are enabled.

        Raises:
            MissingExtractorError: When ``extract_patch_stats`` is enabled but
                no extractor was supplied.
        """
        features: dict[str, Any] = {}
        p = _pfx(prefix)
        sequence = observation.sequence

        features[p("resolution")] = sequence.resolution
        features[p("crs")] = sequence.crs
        features[p("ndvi_count")] = _num(len(sequence.ndvi_paths))
        features[p("evi_count")] = _num(len(sequence.evi_paths))
        features[p("pair_count")] = _num(len(sequence.pairs))
        features[p("paired_dates")] = _num(
            sum(1 for pair in sequence.pairs if pair.ndvi is not None and pair.evi is not None)
        )
        features[p("missing_ndvi_dates")] = _num(
            sum(1 for pair in sequence.pairs if pair.ndvi is None)
        )
        features[p("missing_evi_dates")] = _num(
            sum(1 for pair in sequence.pairs if pair.evi is None)
        )
        gaps = [float(g) for g in sequence.gap_days if g is not None]
        features[p("gap_count")] = _num(len(gaps))
        features[p("max_gap_days")] = max(gaps) if gaps else None
        features[p("mean_gap_days")] = _num(sum(gaps) / len(gaps)) if gaps else None
        dates = observation.temporal.observation_dates or sequence.sorted_dates
        if dates:
            features[p("span_days")] = _num((max(dates) - min(dates)).days)
            features[p("first_date")] = min(dates).isoformat()
            features[p("last_date")] = max(dates).isoformat()
        else:
            features[p("span_days")] = None
            features[p("first_date")] = None
            features[p("last_date")] = None
        features[p("has_pairs")] = int((features[p("pair_count")] or 0) > 0)

        if self.config.extract_patch_stats:
            if extractor is None:
                raise MissingExtractorError(
                    "extract_patch_stats requires an extractor "
                    "(e.g. stam.get_patch)",
                )
            features.update(
                self._patch_features(observation, extractor=extractor, prefix=prefix)
            )
        return features

    # -- Patch statistics ----------------------------------------------------- #

    def _patch_features(
        self,
        observation: Any,
        *,
        extractor: Callable[..., Any],
        prefix: str,
    ) -> dict[str, Any]:
        p = _pfx(prefix)
        out: dict[str, Any] = {}
        lon = observation.location.lon
        lat = observation.location.lat
        size = self.config.patch_size
        used = 0
        for pair in observation.sequence.pairs:
            if used >= self.config.max_dates:
                break
            if pair.ndvi is None or pair.evi is None:
                continue
            for index in ("ndvi", "evi"):
                record = getattr(pair, index)
                patch = extractor(record.path, lon, lat, size=size)
                summary = _patch_summary(patch)
                for stat in _PATCH_STATS:
                    key = p(f"d{used}.{index}.{stat}")
                    out[key] = summary.get(stat)
            used += 1
        out[p("patch_dates_used")] = _num(used)
        return out


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _pfx(prefix: str) -> Any:
    def wrap(key: str) -> str:
        return f"{prefix}.{key}" if prefix else key

    return wrap


def _num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _patch_summary(patch: Any) -> dict[str, float]:
    """Mean / std / min / max over the real (non-padded) patch pixels."""
    array = getattr(patch, "array", None)
    if array is None:
        return {}
    mask = getattr(patch, "mask", None)
    values = array[mask] if mask is not None else array.ravel()
    if values.size == 0:
        return {}
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }
