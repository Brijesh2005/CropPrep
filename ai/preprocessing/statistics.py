"""Dataset statistics: summaries, distributions and reports.

:class:`DatasetStatistics` computes a :class:`StatisticsReport` over a set of
observations — class distribution, yield distribution, sequence-length
distribution, missing values, numeric feature statistics and (optionally)
patch statistics — and persists it as JSON.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .logger import get_logger

logger = get_logger("statistics")


@dataclass(slots=True)
class StatisticsReport:
    """Immutable summary of a dataset."""

    total_observations: int
    class_distribution: dict[str, int] = field(default_factory=dict)
    yield_distribution: dict[str, float] = field(default_factory=dict)
    sequence_length_distribution: dict[str, float] = field(default_factory=dict)
    missing_values: dict[str, int] = field(default_factory=dict)
    feature_statistics: dict[str, dict[str, float]] = field(default_factory=dict)
    patch_statistics: dict[str, Any] = field(default_factory=dict)
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, directory: str | Path) -> Path:
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "dataset_statistics.json"
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Statistics report saved", extra={"path": str(path)})
        return path


class DatasetStatistics:
    """Compute statistics over observations."""

    @classmethod
    def summarize(
        cls,
        observations: Sequence[Any],
        *,
        extractor: Any | None = None,
        patch_size: int = 128,
        sample_cap: int = 20,
    ) -> StatisticsReport:
        """Compute a full statistics report.

        Args:
            observations: Accepted observations.
            extractor: Optional patch extractor for patch statistics.
            patch_size: Patch edge used by the extractor.
            sample_cap: Cap on observations sampled for patch statistics.
        """
        obs_list = list(observations)
        report = StatisticsReport(total_observations=len(obs_list))

        crops: list[str] = []
        yields: list[float] = []
        lengths: list[int] = []
        missing: dict[str, int] = {}
        numeric: dict[str, list[float]] = {}

        for obs in obs_list:
            crops.append(str(obs.crop) if obs.crop is not None else "<none>")
            if obs.yield_value is not None:
                yields.append(float(obs.yield_value))
            lengths.append(obs.num_observations())

            fields = dict(obs.tabular.fields) if obs.tabular else {}
            for key, value in fields.items():
                if value is None or value == "" or _is_nan(value):
                    missing[key] = missing.get(key, 0) + 1
                if _is_numeric(value):
                    numeric.setdefault(key, []).append(float(value))

        report.class_distribution = _counts(crops)
        report.yield_distribution = _summary(yields)
        report.sequence_length_distribution = {
            "min": float(min(lengths)) if lengths else 0.0,
            "max": float(max(lengths)) if lengths else 0.0,
            "mean": float(np.mean(lengths)) if lengths else 0.0,
            "count": float(len(lengths)),
        }
        report.missing_values = missing
        report.feature_statistics = {
            key: _summary(values) for key, values in numeric.items()
        }

        if extractor is not None:
            report.patch_statistics = cls._patch_statistics(
                obs_list, extractor, patch_size, sample_cap
            )
        return report

    @classmethod
    def _patch_statistics(
        cls, observations: Sequence[Any], extractor: Any, patch_size: int, cap: int
    ) -> dict[str, Any]:
        valid_ratios: list[float] = []
        sizes: list[tuple[int, int]] = []
        for obs in observations[:cap]:
            for pair in obs.sequence.pairs[:2]:
                path = (pair.ndvi or pair.evi)
                if path is None:
                    continue
                patch = extractor(path.path, obs.location.lon, obs.location.lat, size=patch_size)
                valid_ratios.append(patch.valid_ratio)
                sizes.append((patch.shape[0], patch.shape[1]))
        return {
            "sampled": len(valid_ratios),
            "valid_ratio": _summary(valid_ratios),
            "shapes": [list(s) for s in dict.fromkeys(sizes)],
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _counts(values: list) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: -item[1]))


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0.0}
    array = np.asarray(values, dtype="float64")
    return {
        "count": float(len(array)),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()) if len(array) > 1 else 0.0,
    }


def _is_numeric(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return True
    if isinstance(value, str):
        try:
            float(value)
            return True
        except ValueError:
            return False
    return False


def _is_nan(value: Any) -> bool:
    try:
        return bool(np.isnan(float(value)))
    except (TypeError, ValueError):
        return False
