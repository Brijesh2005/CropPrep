"""Quality-control reports over generated training samples (R2.3).

:class:`~training.quality.samples.report.SampleQualityReport` aggregates a
resolved :class:`~training.stam.observation_resolver.ObservationCorpus` into
a single JSON report: acceptance / rejection / error counts, the dominant STAM
issue codes and severities, the quality-score distribution of accepted cells,
and per-crop / per-year / per-season acceptance rates.

Typical usage::

    from training.quality.samples import build_report

    report = build_report(corpus, output_dir="data/out/qc")
"""

from __future__ import annotations

from .exceptions import SampleQualityConfigError, SampleQualityError
from .report import SampleQualityReport, build_report

__all__ = [
    "SampleQualityConfigError",
    "SampleQualityError",
    "SampleQualityReport",
    "build_report",
]
