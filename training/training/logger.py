"""Experiment logger — CSV / JSON metrics, config snapshot, git hash.

Writes, under the run directory:

* ``metrics.csv`` — one row per epoch (union of all metric keys),
* ``metrics.json`` — per-epoch records plus the final summary,
* ``config.yaml`` — resolved training configuration snapshot,
* ``model_config.yaml`` / ``preprocessing_config.yaml`` — model / data configs,
* ``environment.json`` — python / torch / hardware info,
* ``git.json`` — commit hash + branch (best-effort).

All writers are created lazily and tolerate missing optional dependencies.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Mapping

from .utils import get_environment_info, get_git_branch, get_git_hash, is_primary

from shared.utils import yaml_safe


class ExperimentLogger:
    """Structured metrics + artifact logger for a single run."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        name: str = "cropfusion",
        csv_enabled: bool = True,
        json_enabled: bool = True,
        console_level: str = "INFO",
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.csv_enabled = csv_enabled
        self.json_enabled = json_enabled

        self._csv_path = self.run_dir / "metrics.csv"
        self._json_path = self.run_dir / "metrics.json"
        self._records: list[dict[str, Any]] = []

        self._logger = logging.getLogger(f"cropfusion.{name}")
        self._logger.setLevel(console_level.upper())

    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #

    def log_epoch(self, epoch: int, metrics: Mapping[str, Any]) -> None:
        """Append one epoch row to CSV + JSON."""
        record = {"epoch": int(epoch)}
        record.update({str(k): v for k, v in metrics.items()})
        self._records.append(record)
        if is_primary():
            if self.csv_enabled:
                self._rewrite_csv()
            if self.json_enabled:
                self._rewrite_json()

    def log_metrics(self, step: int, metrics: Mapping[str, Any]) -> None:
        """Record a non-epoch metric snapshot (train steps)."""
        record = {"step": int(step)}
        record.update({str(k): v for k, v in metrics.items()})
        self._records.append(record)

    def finalize(self, summary: Mapping[str, Any] | None = None) -> Path:
        """Write the final JSON report (per-epoch records + summary)."""
        payload = {
            "name": self.name,
            "records": self._records,
            "summary": dict(summary or {}),
        }
        self._json_path.write_text(
            json.dumps(payload, indent=2, default=_json_default), encoding="utf-8"
        )
        return self._json_path

    # ------------------------------------------------------------------ #
    # Snapshots
    # ------------------------------------------------------------------ #

    def save_config_snapshot(
        self,
        training_config: Any | None = None,
        model_config: Any | None = None,
        preprocessing_config: Any | None = None,
    ) -> Path:
        """Persist resolved configs as YAML under the run directory."""
        import yaml

        data: dict[str, Any] = {}
        if training_config is not None:
            data["training"] = training_config.model_dump()
        if model_config is not None:
            data["model"] = model_config.model_dump()
        if preprocessing_config is not None:
            data["preprocessing"] = preprocessing_config.model_dump()
        path = self.run_dir / "config.yaml"
        path.write_text(
            yaml.safe_dump(yaml_safe(data), sort_keys=False), encoding="utf-8"
        )
        return path

    def save_environment(self) -> Path:
        path = self.run_dir / "environment.json"
        path.write_text(
            json.dumps(get_environment_info(), indent=2), encoding="utf-8"
        )
        return path

    def save_git(self) -> Path:
        hash_value = get_git_hash()
        path = self.run_dir / "git.json"
        path.write_text(
            json.dumps(
                {"commit": hash_value, "branch": get_git_branch(),
                 "dirty": hash_value is None},
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    # ------------------------------------------------------------------ #
    # Console
    # ------------------------------------------------------------------ #

    def info(self, message: str, **extra: Any) -> None:
        if is_primary():
            self._logger.info("%s %s", message, _fmt_extra(extra))

    def warning(self, message: str, **extra: Any) -> None:
        self._logger.warning("%s %s", message, _fmt_extra(extra))

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _rewrite_csv(self) -> None:
        """Rewrite metrics.csv from the accumulated records (epochs are few)."""
        columns: list[str] = []
        for record in self._records:
            for key in record:
                if key not in columns:
                    columns.append(key)
        with self._csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for record in self._records:
                writer.writerow({key: record.get(key, "") for key in columns})

    def _rewrite_json(self) -> None:
        """Keep metrics.json up to date as training progresses."""
        self.finalize()


def _fmt_extra(extra: Mapping[str, Any]) -> str:
    if not extra:
        return ""
    return " ".join(f"{key}={value}" for key, value in extra.items())


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)
