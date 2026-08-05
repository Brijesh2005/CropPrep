"""MLOps settings - environment driven (12-factor), mirroring the other
CropFusion packages (``MLOPS_`` prefix)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

from pydantic import BaseModel, Field

ENV_PREFIX = "MLOPS_"


class MLOpsSettings(BaseModel):
    """Settings for the model registry and scheduler."""

    model_config = {"extra": "forbid"}

    #: Root directory for the filesystem model registry.
    registry_dir: Path = Path("models/registry")
    #: Directory for generated reports (drift, fairness, benchmarks, releases).
    reports_dir: Path = Path("reports")
    #: Directory for experiment tracking files.
    experiments_dir: Path = Path("experiments")
    #: Backups directory (used by scheduler reminder).
    backups_dir: Path = Path("backups")

    #: Interval (seconds) between scheduler cycles.
    interval_seconds: int = Field(default=3600, ge=30)
    #: Minimum accuracy accepted by the promote gate (0-1).
    min_accuracy: float = Field(default=0.80, ge=0.0, le=1.0)
    #: Maximum regression allowed vs the incumbent on the benchmark gate (%). 
    max_latency_regression_pct: float = Field(default=10.0, ge=0.0)
    #: Production models to keep around before archiving.
    keep_production_versions: int = Field(default=3, ge=1)

    #: Reference dataset for the drift gate (parquet/csv) - optional.
    drift_reference_data: Path | None = None
    #: Feature columns used by the drift gate.
    drift_feature_columns: list[str] = Field(default_factory=list)
    #: Label column used by the drift gate.
    drift_label_column: str | None = None

    #: Endpoint of a Prometheus pushgateway to publish ML-QA verdicts (optional).
    pushgateway: str | None = None


def load_settings(
    env: Mapping[str, str] | None = None,
    file: str | Path | None = None,
) -> MLOpsSettings:
    """Load settings from ``env`` (default: ``os.environ``) then YAML ``file``."""
    raw: dict[str, str] = dict(os.environ if env is None else env)

    if file and Path(file).exists():
        import yaml

        raw.update({str(k): str(v) for k, v in yaml.safe_load(Path(file).read_text()).items()})

    kwargs: dict[str, object] = {}
    for key, value in raw.items():
        if key.startswith(ENV_PREFIX):
            field = key[len(ENV_PREFIX):].lower()
            kwargs[field] = value
    return MLOpsSettings.model_validate(kwargs)
