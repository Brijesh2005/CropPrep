"""Corpus-level dataset statistics.

:class:`CorpusStatistics` summarises an
:class:`~training.stam.observation_resolver.ObservationCorpus` — the R2.3
training dataset — across every useful axis: sample status, crop, year,
season, administrative area, location and quality score distribution. It is
the input to balancing decisions (:mod:`training.feature_engineering.balancing`)
and the quality-control reports (:mod:`training.quality.samples`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .logger import get_logger
from .utils import group_counts, observations_from_corpus

logger = get_logger("statistics")


@dataclass
class CorpusStatistics:
    """Aggregate statistics for a resolved sample corpus.

    Classifiers use :attr:`crop` for class balancing; ``yield`` statistics are
    computed over accepted observations carrying a numeric yield label.
    """

    total: int = 0
    status_counts: dict[str, int] = field(default_factory=dict)
    accepted_count: int = 0
    rejected_count: int = 0
    error_count: int = 0
    quality: dict[str, Any] = field(default_factory=dict)
    by_crop: dict[str, int] = field(default_factory=dict)
    by_year: dict[str, int] = field(default_factory=dict)
    by_season: dict[str, int] = field(default_factory=dict)
    by_district: dict[str, int] = field(default_factory=dict)
    by_location: dict[str, int] = field(default_factory=dict)
    yield_stats: dict[str, Any] = field(default_factory=dict)
    missing_labels: dict[str, int] = field(default_factory=dict)

    # -- Builders ------------------------------------------------------------- #

    @classmethod
    def summarize(cls, corpus: Any) -> "CorpusStatistics":
        """Compute statistics from a corpus (or a list of observations)."""
        has_samples = hasattr(corpus, "samples") and not isinstance(corpus, (list, tuple))
        samples = list(corpus.samples) if has_samples else []
        accepted = observations_from_corpus(corpus)

        if has_samples:
            accepted_samples = [s for s in samples if s.status == "accepted"]
            status_counts: dict[str, int] = {"accepted": 0, "rejected": 0, "error": 0}
            for sample in samples:
                status_counts[sample.status] += 1
            quality_scores = [
                float(s.quality_score)
                for s in accepted_samples
                if s.quality_score is not None
            ]
            total = len(samples)
        else:
            status_counts = {"accepted": len(accepted), "rejected": 0, "error": 0}
            quality_scores = [
                float(obs.quality.overall_score)
                for obs in accepted
                if obs.quality is not None
            ]
            total = len(accepted)

        missing_crop = sum(1 for o in accepted if o.crop is None)
        missing_yield = sum(1 for o in accepted if o.yield_value is None)

        return cls(
            total=total,
            status_counts=status_counts,
            accepted_count=status_counts.get("accepted", 0),
            rejected_count=status_counts.get("rejected", 0),
            error_count=status_counts.get("error", 0),
            quality=_score_summary(quality_scores),
            by_crop=group_counts(accepted, "crop"),
            by_year=_year_counts(accepted),
            by_season=_season_counts(accepted),
            by_district=_district_counts(accepted),
            by_location=_location_counts(accepted),
            yield_stats=_yield_summary(accepted),
            missing_labels={"crop": missing_crop, "yield": missing_yield},
        )

    # -- Output --------------------------------------------------------------- #

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "status": self.status_counts,
            "accepted": self.accepted_count,
            "rejected": self.rejected_count,
            "errors": self.error_count,
            "acceptance_rate": round(
                self.accepted_count / self.total, 4
            ) if self.total else 0.0,
            "quality": self.quality,
            "by_crop": self.by_crop,
            "by_year": self.by_year,
            "by_season": self.by_season,
            "by_district": self.by_district,
            "by_location": self.by_location,
            "yield": self.yield_stats,
            "missing_labels": self.missing_labels,
        }

    def save(self, output_dir: str | Path) -> Path:
        """Write ``dataset_statistics.json`` and return its path."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "dataset_statistics.json"
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    def to_frame(self) -> Any:
        """A ``(crop, count)`` DataFrame for quick plotting / inspection."""
        import pandas as pd

        return pd.DataFrame(
            sorted(self.by_crop.items(), key=lambda kv: -kv[1]),
            columns=["crop", "count"],
        )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _score_summary(scores: list[float]) -> dict[str, Any]:
    if not scores:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None,
                "std": None, "q25": None, "q75": None}
    ordered = sorted(scores)
    count = len(ordered)
    median = ordered[count // 2] if count % 2 else (
        (ordered[count // 2 - 1] + ordered[count // 2]) / 2.0
    )
    mean = sum(ordered) / count
    variance = sum((x - mean) ** 2 for x in ordered) / count
    return {
        "count": count,
        "min": round(ordered[0], 2),
        "max": round(ordered[-1], 2),
        "mean": round(mean, 2),
        "median": round(median, 2),
        "std": round(variance ** 0.5, 2),
        "q25": round(ordered[int(count * 0.25)], 2),
        "q75": round(ordered[min(count - 1, int(count * 0.75))], 2),
    }


def _year_counts(observations: Sequence[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for obs in observations:
        year = getattr(getattr(obs, "temporal", None), "year", None)
        if year is None:
            continue
        key = str(year)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _season_counts(observations: Sequence[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for obs in observations:
        season = getattr(getattr(obs, "temporal", None), "season", None)
        if season is None:
            continue
        counts[str(season)] = counts.get(str(season), 0) + 1
    return counts


def _district_counts(observations: Sequence[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for obs in observations:
        admin = getattr(getattr(obs, "location", None), "admin", None)
        district = getattr(admin, "district", None)
        if district is None:
            continue
        counts[str(district)] = counts.get(str(district), 0) + 1
    return counts


def _location_counts(observations: Sequence[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for obs in observations:
        name = getattr(getattr(obs, "location", None), "dataset_location_name", None)
        if name is None:
            continue
        counts[str(name)] = counts.get(str(name), 0) + 1
    return counts


def _yield_summary(observations: Sequence[Any]) -> dict[str, Any]:
    values = [
        float(obs.yield_value)
        for obs in observations
        if obs.yield_value is not None
    ]
    return _score_summary(values)
