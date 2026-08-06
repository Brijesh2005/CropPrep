"""Quality-control reports over the generated training-sample corpus.

:class:`SampleQualityReport` aggregates the per-cell outcomes of an
:class:`~training.stam.observation_resolver.ObservationCorpus` into one
JSON report that answers:

* how many cells were accepted / rejected / errored (and why),
* which STAM quality issue codes dominate the accepted observations,
* the severity histogram across those issues,
* the quality-score distribution of accepted samples,
* acceptance rates per crop / year / season (where the data is weakest),
* the top failure codes among error cells.

The report only reads the corpus; it never re-runs STAM. :func:`build_report`
writes ``sample_quality_report.json`` and returns the parsed payload.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .exceptions import SampleQualityError
from .logger import get_logger

logger = get_logger("samples")


@dataclass
class SampleQualityReport:
    """Aggregated quality-control report for a resolved corpus."""

    status_counts: dict[str, int] = field(default_factory=dict)
    total: int = 0
    acceptance_rate: float = 0.0
    quality_score: dict[str, Any] = field(default_factory=dict)
    issue_codes: dict[str, int] = field(default_factory=dict)
    severity_counts: dict[str, int] = field(default_factory=dict)
    by_crop: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_year: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_season: dict[str, dict[str, Any]] = field(default_factory=dict)
    top_error_codes: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_corpus(cls, corpus: Any) -> "SampleQualityReport":
        """Compute the report from a corpus (or list of resolved samples)."""
        samples = _samples(corpus)
        if not samples:
            raise SampleQualityError("No samples to report on")

        status_counts: dict[str, int] = {"accepted": 0, "rejected": 0, "error": 0}
        for sample in samples:
            status_counts[sample.status] += 1
        total = len(samples)
        accepted = [s for s in samples if s.status == "accepted"]

        scores = [
            float(s.quality_score)
            for s in accepted
            if s.quality_score is not None
        ]
        quality_score = _score_summary(scores)

        issue_codes: dict[str, int] = {}
        severity_counts: dict[str, int] = {}
        for sample in accepted:
            observation = getattr(sample, "observation", None)
            report = getattr(observation, "quality", None)
            if report is None:
                continue
            for issue in getattr(report, "issues", []):
                code = issue.code
                issue_codes[code] = issue_codes.get(code, 0) + 1
                severity_counts[issue.severity] = severity_counts.get(issue.severity, 0) + 1

        by_crop = _acceptance_groups(accepted, _crop_of)
        by_year = _acceptance_groups(accepted, _year_of)
        by_season = _acceptance_groups(accepted, _season_of)

        error_codes: dict[str, int] = {}
        for sample in samples:
            if sample.status != "error":
                continue
            error = getattr(sample, "error", None) or {}
            code = error.get("code", "unknown")
            error_codes[code] = error_codes.get(code, 0) + 1

        report = cls(
            status_counts=status_counts,
            total=total,
            acceptance_rate=round(status_counts["accepted"] / total, 4) if total else 0.0,
            quality_score=quality_score,
            issue_codes=issue_codes,
            severity_counts=severity_counts,
            by_crop=by_crop,
            by_year=by_year,
            by_season=by_season,
            top_error_codes=dict(sorted(error_codes.items(), key=lambda kv: -kv[1])),
        )
        logger.info(
            "Sample quality report built",
            extra={"total": total, **status_counts},
        )
        return report

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "status": self.status_counts,
            "acceptance_rate": self.acceptance_rate,
            "quality_score": self.quality_score,
            "issue_codes": self.issue_codes,
            "severity_counts": self.severity_counts,
            "by_crop": self.by_crop,
            "by_year": self.by_year,
            "by_season": self.by_season,
            "top_error_codes": self.top_error_codes,
        }

    def save(self, output_dir: str | Path) -> Path:
        """Write ``sample_quality_report.json`` and return its path."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "sample_quality_report.json"
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


def build_report(corpus: Any, output_dir: str | Path | None = None) -> SampleQualityReport:
    """Build a :class:`SampleQualityReport` for a corpus.

    Args:
        corpus: An :class:`ObservationCorpus` or list of resolved samples.
        output_dir: When given, the report is also written as JSON.

    Raises:
        SampleQualityError: When the corpus has no samples.
    """
    report = SampleQualityReport.from_corpus(corpus)
    if output_dir is not None:
        report.save(output_dir)
    return report


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _samples(corpus: Any) -> list[Any]:
    if corpus is None:
        return []
    if hasattr(corpus, "samples"):
        return list(corpus.samples)
    return list(corpus)


def _score_summary(scores: list[float]) -> dict[str, Any]:
    if not scores:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    ordered = sorted(scores)
    count = len(ordered)
    median = ordered[count // 2] if count % 2 else (
        (ordered[count // 2 - 1] + ordered[count // 2]) / 2.0
    )
    return {
        "count": count,
        "min": round(ordered[0], 2),
        "max": round(ordered[-1], 2),
        "mean": round(sum(ordered) / count, 2),
        "median": round(median, 2),
    }


def _acceptance_groups(
    accepted: list[Any], key_of: Any
) -> dict[str, dict[str, Any]]:
    """Per-key ``{accepted, total, rate}`` over the accepted sample list."""
    total_by_key: dict[str, int] = {}
    accepted_by_key: dict[str, int] = {}
    for sample in accepted:
        key = key_of(sample)
        if key is None:
            continue
        total_by_key[key] = total_by_key.get(key, 0) + 1
        accepted_by_key[key] = accepted_by_key.get(key, 0) + 1
    out: dict[str, dict[str, Any]] = {}
    for key, total in total_by_key.items():
        out[key] = {
            "total": total,
            "accepted": accepted_by_key.get(key, 0),
            "rate": round(accepted_by_key.get(key, 0) / total, 4),
        }
    return out


def _crop_of(sample: Any) -> str | None:
    obs = getattr(sample, "observation", None)
    crop = getattr(obs, "crop", None)
    return str(crop) if crop else None


def _year_of(sample: Any) -> str | None:
    obs = getattr(sample, "observation", None)
    temporal = getattr(obs, "temporal", None)
    year = getattr(temporal, "year", None)
    return str(year) if year is not None else None


def _season_of(sample: Any) -> str | None:
    obs = getattr(sample, "observation", None)
    temporal = getattr(obs, "temporal", None)
    season = getattr(temporal, "season", None)
    return str(season) if season else None
