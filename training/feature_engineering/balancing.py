"""Class-balancing analysis over the generated training corpus.

:class:`BalancingReport` quantifies crop-class imbalance in the accepted
samples and derives concrete balancing artefacts:

* per-class counts and shares,
* the imbalance ratio (``max / min`` class frequency),
* minority / majority class lists,
* inverse-frequency :meth:`class_weights` (normalised) and per-sample weights,
* a :attr:`recommended_strategy` for downstream training.

Balancing is purely statistical: it never mutates the corpus. The weights feed
the Phase 4 loss / sampler decisions at training time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .logger import get_logger
from .utils import observations_from_corpus

logger = get_logger("balancing")

MIN_SAMPLES_FOR_STRATEGY = 2


@dataclass
class BalancingReport:
    """Class-balance analysis of the accepted samples."""

    class_counts: dict[str, int] = field(default_factory=dict)
    class_shares: dict[str, float] = field(default_factory=dict)
    total: int = 0
    n_classes: int = 0
    imbalance_ratio: float | None = None
    minority_classes: list[str] = field(default_factory=list)
    majority_classes: list[str] = field(default_factory=list)
    recommended_strategy: str = "balanced"
    by_year: dict[str, int] = field(default_factory=dict)
    by_season: dict[str, int] = field(default_factory=dict)

    # -- Builders ------------------------------------------------------------- #

    @classmethod
    def summarize(cls, corpus: Any) -> "BalancingReport":
        """Analyse class balance over a corpus (or list of observations)."""
        accepted = observations_from_corpus(corpus)
        counts: dict[str, int] = {}
        by_year: dict[str, int] = {}
        by_season: dict[str, int] = {}
        for obs in accepted:
            crop = str(obs.crop or "unknown")
            counts[crop] = counts.get(crop, 0) + 1
            year = getattr(getattr(obs, "temporal", None), "year", None)
            if year is not None:
                key = str(year)
                by_year[key] = by_year.get(key, 0) + 1
            season = getattr(getattr(obs, "temporal", None), "season", None)
            if season is not None:
                key = str(season)
                by_season[key] = by_season.get(key, 0) + 1

        total = sum(counts.values())
        n_classes = len(counts)
        shares = {k: round(v / total, 4) for k, v in counts.items()} if total else {}

        frequencies = sorted(counts.values())
        imbalance = None
        if len(frequencies) >= MIN_SAMPLES_FOR_STRATEGY and frequencies and frequencies[0] > 0:
            imbalance = round(frequencies[-1] / frequencies[0], 4)

        mean = total / n_classes if n_classes else 0.0
        minority = [k for k, v in counts.items() if v < mean]
        majority = [k for k, v in counts.items() if v > mean]

        strategy = "balanced"
        if imbalance is not None and imbalance >= 2.0:
            strategy = "oversample_minority"
        if imbalance is not None and imbalance >= 5.0:
            strategy = "combined_weights_and_oversample"

        return cls(
            class_counts=counts,
            class_shares=shares,
            total=total,
            n_classes=n_classes,
            imbalance_ratio=imbalance,
            minority_classes=sorted(minority),
            majority_classes=sorted(majority),
            recommended_strategy=strategy,
            by_year=by_year,
            by_season=by_season,
        )

    # -- Balancing artefacts -------------------------------------------------- #

    def class_weights(self) -> dict[str, float]:
        """Inverse-frequency class weights (normalised to sum 1)."""
        if not self.class_counts:
            return {}
        total = self.total
        raw = {
            k: (total / (v * self.n_classes)) if v else 0.0
            for k, v in self.class_counts.items()
        }
        scale = sum(raw.values()) or 1.0
        return {k: round(v / scale, 6) for k, v in raw.items()}

    def sample_weight(self, crop: str | None) -> float:
        """Per-sample weight for a crop class (inverse frequency)."""
        return self.class_weights().get(str(crop or "unknown"), 1.0)

    def sample_weights(self, observations: Sequence[Any]) -> list[float]:
        """Per-sample weights for a list of observations (ordered)."""
        weights = self.class_weights()
        return [weights.get(str(obs.crop or "unknown"), 1.0) for obs in observations]

    # -- Output --------------------------------------------------------------- #

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "n_classes": self.n_classes,
            "imbalance_ratio": self.imbalance_ratio,
            "class_counts": self.class_counts,
            "class_shares": self.class_shares,
            "class_weights": self.class_weights(),
            "minority_classes": self.minority_classes,
            "majority_classes": self.majority_classes,
            "recommended_strategy": self.recommended_strategy,
            "by_year": self.by_year,
            "by_season": self.by_season,
        }

    def save(self, output_dir: str | Path) -> Path:
        """Write ``balancing_report.json`` and return its path."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "balancing_report.json"
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path
