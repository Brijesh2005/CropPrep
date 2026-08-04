"""Experiment tracking - append-only JSONL run log.

Each training run is recorded with its configuration, metrics and provenance so
results are reproducible and comparable across runs. No external service is
required; files live under ``<experiments_dir>/runs.jsonl``.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import MLOpsSettings


class ExperimentTracker:
    """Append-only experiment run log."""

    def __init__(self, settings: MLOpsSettings) -> None:
        self.dir = Path(settings.experiments_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._path = self.dir / "runs.jsonl"

    def log(
        self,
        *,
        model_name: str,
        config: dict[str, Any],
        metrics: dict[str, float],
        dataset_version: str | None = None,
        git_commit: str | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        """Append one run and return the recorded payload."""
        run = {
            "run_id": uuid.uuid4().hex[:12],
            "model_name": model_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": config,
            "metrics": metrics,
            "dataset_version": dataset_version,
            "git_commit": git_commit,
            "notes": notes,
        }
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(run, default=str) + "\n")
        return run

    def runs(self, model_name: str | None = None) -> list[dict[str, Any]]:
        if not self._path.exists():
            return []
        runs = [
            json.loads(line)
            for line in self._path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if model_name:
            runs = [r for r in runs if r.get("model_name") == model_name]
        return sorted(runs, key=lambda r: r.get("timestamp", ""))

    def best(
        self, model_name: str | None = None, metric: str = "accuracy"
    ) -> dict[str, Any] | None:
        """The run with the highest ``metric`` value."""
        scored = [
            r for r in self.runs(model_name) if r.get("metrics", {}).get(metric) is not None
        ]
        if not scored:
            return None
        return max(scored, key=lambda r: float(r["metrics"][metric]))

    def export(self, out_path: str | Path) -> Path:
        runs = self.runs()
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(runs, indent=2, default=str), encoding="utf-8")
        return out
