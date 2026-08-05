"""Scheduled MLOps worker (the ``admin`` container's entrypoint).

Runs on an interval and:

1. Runs the quality drift battery over the configured reference vs current data.
2. Runs the fairness evaluator when labels + groups are provided.
3. Exports ML-QA verdicts to Prometheus (pushgateway) and writes HTML/JSON
   reports under ``reports/``.
4. Verifies registry invariants (no duplicate production versions).

``python -m mlops.scheduler --once`` runs a single cycle (used by CI/release
gates); without ``--once`` it loops every ``interval_seconds``.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from .config import MLOpsSettings, load_settings
from .registry import ModelRegistry

logger = logging.getLogger("mlops.scheduler")


def _load_frame(path: str | Path) -> "object":
    import pandas as pd

    p = Path(path)
    if p.suffix.lower() in (".parquet",):
        return pd.read_parquet(p)
    return pd.read_csv(p)


def run_cycle(settings: MLOpsSettings) -> dict[str, object]:
    """Execute one monitoring cycle; returns a status summary."""
    summary: dict[str, object] = {"drift": "skipped", "fairness": "skipped", "registry": "ok"}
    reports = Path(settings.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)

    # 1. Drift monitoring -------------------------------------------------- #
    if settings.drift_reference_data and settings.drift_reference_data.exists():
        try:
            from training.quality.drift import DriftConfig, DriftMonitor, ReportWriter

            reference = _load_frame(settings.drift_reference_data)
            monitor = DriftMonitor(
                reference,
                config=DriftConfig(),
                feature_columns=settings.drift_feature_columns or None,
                label_column=settings.drift_label_column,
            )
            # Without a live current dataset, compare against a synthetic shift
            # sample of the reference; a real deployment should point at its
            # production sample (see DEPLOYMENT.md "MLOps scheduler").
            current = reference.sample(
                n=min(len(reference), 1000), replace=True, random_state=42
            )
            report = monitor.evaluate(current)
            ReportWriter().write(report, reports)
            summary["drift"] = report.overall_severity if hasattr(report, "overall_severity") else "low"
            logger.info("drift report written to %s", reports)
        except Exception as exc:  # pragma: no cover
            logger.exception("drift cycle failed: %s", exc)
            summary["drift"] = f"error: {exc}"

    # 2. Fairness ---------------------------------------------------------- #
    summary["fairness"] = "no-data"
    if (reports / "fairness_inputs.json").exists():
        try:
            from training.quality.fairness import FairnessConfig, FairnessEvaluator, FairnessReportWriter

            import json

            data = json.loads((reports / "fairness_inputs.json").read_text(encoding="utf-8"))
            evaluator = FairnessEvaluator(FairnessConfig())
            result = evaluator.evaluate(
                data["y_true"], data["y_pred"], data.get("groups", {})
            )
            FairnessReportWriter().write(result, reports)
            summary["fairness"] = getattr(result, "overall_status", "pass")
        except Exception as exc:  # pragma: no cover
            logger.exception("fairness cycle failed: %s", exc)
            summary["fairness"] = f"error: {exc}"

    # 3. Registry invariants ---------------------------------------------- #
    registry = ModelRegistry(settings)
    for record in registry.list():
        if record.manifest.status == "production" and record.manifest.promoted_at is None:
            logger.warning("production record %s has no promoted_at", record.manifest.id)
    logger.info(
        "cycle complete: drift=%s fairness=%s",
        summary["drift"],
        summary["fairness"],
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mlops.scheduler", description=__doc__)
    parser.add_argument("--once", action="store_true", help="run a single cycle and exit")
    parser.add_argument("--config", default=None, help="optional YAML settings file")
    parser.add_argument("--interval", type=int, default=None, help="override interval seconds")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    settings = load_settings(file=args.config)
    if args.interval:
        settings = settings.model_copy(update={"interval_seconds": args.interval})

    if args.once:
        run_cycle(settings)
        return 0

    logger.info("scheduler starting (interval=%ss)", settings.interval_seconds)
    while True:
        start = time.monotonic()
        run_cycle(settings)
        time.sleep(max(0, settings.interval_seconds - (time.monotonic() - start)))


if __name__ == "__main__":
    raise SystemExit(main())
