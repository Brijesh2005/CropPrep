"""Configuration for the evaluation package (Phase R5).

Everything is configurable through YAML (or ``EVAL_*`` env vars), mirroring the
resolution order of the other CropFusion packages:

    env (``EVAL_<SECTION>__<KEY>``) > YAML (``EVAL_CONFIG_FILE``) > defaults

Every field is validated by pydantic. The root :class:`EvaluationConfig`
contains one section per subsystem (general, metrics, comparison, ablation,
error_analysis).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field

from shared.config import apply_case_insensitive, deep_merge, parse_env
from shared.utils import yaml_safe

from .exceptions import EvaluationConfigurationError

ENV_PREFIX = "EVAL_"


class GeneralConfig(BaseModel):
    """Device / seed / batching defaults for evaluation runs."""

    model_config = ConfigDict(extra="forbid")

    #: ``auto`` (best available) | ``cpu`` | ``cuda``.
    device: str = "auto"
    seed: int = 42
    batch_size: int = Field(default=32, ge=1)
    num_workers: int = Field(default=0, ge=0)
    #: Collect shared representations for error analysis / explainability.
    collect_embeddings: bool = True
    #: Default output directory for reports.
    output_dir: str = "artifacts/evaluation"


class MetricsConfig(BaseModel):
    """Extended metric settings (beyond the training-time metrics)."""

    model_config = ConfigDict(extra="forbid")

    top_k: int = Field(default=3, ge=1)
    #: macro | micro | weighted (classification aggregation).
    average: str = Field(default="macro", pattern="^(macro|micro|weighted)$")
    roc_auc: bool = True
    #: Compute precision-recall curves (macro AUPRC + per-class curves).
    pr_curves: bool = True
    #: Histogram bin count for prediction-error distributions.
    histogram_bins: int = Field(default=10, ge=2)
    #: Percentiles of the absolute-error distribution to report.
    error_percentiles: list[float] = Field(
        default_factory=lambda: [0.05, 0.25, 0.5, 0.75, 0.95]
    )
    #: Fraction of the target magnitude considered "within tolerance".
    tolerance_fraction: float = Field(default=0.1, gt=0.0, le=1.0)


class ComparisonConfig(BaseModel):
    """Comparison table settings."""

    model_config = ConfigDict(extra="forbid")

    top_k_classes: int = Field(default=10, ge=1)
    include_per_class: bool = True
    #: Sort per-class rows by this column.
    sort_by: str = "f1"
    include_per_modality: bool = True


class AblationConfig(BaseModel):
    """Ablation-study settings (variant sweep over a base config)."""

    model_config = ConfigDict(extra="forbid")

    #: Variants to run; ``None`` = all registered variants.
    variants: list[str] | None = None
    #: Metric used to rank the variants (``crop/f1`` by default).
    compare_metric: str = "crop/f1"
    #: max | min — whether higher or lower is better.
    compare_mode: str = Field(default="max", pattern="^(max|min)$")
    #: Inference-speed benchmark iterations (per variant).
    benchmark_iterations: int = Field(default=3, ge=1)
    benchmark_warmup: int = Field(default=1, ge=0)
    seed: int = 42


class ErrorAnalysisConfig(BaseModel):
    """Error-analysis settings (misclassifications / outliers / failures)."""

    model_config = ConfigDict(extra="forbid")

    top_k_errors: int = Field(default=20, ge=1)
    #: Residual percentile above which a regression sample is an outlier.
    outlier_percentile: float = Field(default=0.95, gt=0.0, le=1.0)
    #: Relative absolute error above which a sample is a "failure case".
    failure_relative_error: float = Field(default=0.3, gt=0.0)
    #: Store per-sample details (ids / features / embeddings).
    store_samples: bool = True


class EvaluationConfig(BaseModel):
    """Root evaluation configuration."""

    model_config = ConfigDict(extra="forbid")

    name: str = "cropfusion_evaluation"
    general: GeneralConfig = Field(default_factory=GeneralConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    comparison: ComparisonConfig = Field(default_factory=ComparisonConfig)
    ablation: AblationConfig = Field(default_factory=AblationConfig)
    error_analysis: ErrorAnalysisConfig = Field(default_factory=ErrorAnalysisConfig)

    def to_yaml(self) -> str:
        return yaml.safe_dump(yaml_safe(self.model_dump()), sort_keys=False)

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.write_text(self.to_yaml(), encoding="utf-8")
        return out


def load_evaluation_config(
    config_path: str | Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> EvaluationConfig:
    """Load and validate evaluation settings (env > YAML > defaults)."""
    env_map = dict(os.environ if env is None else env)
    if config_path is None:
        env_config = env_map.get("EVAL_CONFIG_FILE")
        config_path = env_config or None

    data: dict[str, Any] = {}
    if config_path is not None:
        config_file = Path(config_path)
        if not config_file.exists():
            raise EvaluationConfigurationError(
                f"Evaluation config file not found: {config_file}",
                detail=str(config_file),
            )
        try:
            raw = yaml.safe_load(config_file.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise EvaluationConfigurationError(
                f"Malformed evaluation YAML: {exc}", detail=str(config_file)
            ) from exc
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise EvaluationConfigurationError(
                "Evaluation config root must be a mapping"
            )
        data = raw

    parsed_env = parse_env(env_map, prefix=ENV_PREFIX)
    parsed_env.pop("config_file", None)
    merged = deep_merge(data, parsed_env)
    merged = apply_case_insensitive(merged, EvaluationConfig)
    try:
        return EvaluationConfig.model_validate(merged)
    except Exception as exc:  # pydantic.ValidationError
        raise EvaluationConfigurationError(
            f"Invalid evaluation configuration: {exc}"
        ) from exc


def save_evaluation_template(path: str | Path) -> Path:
    """Write an annotated YAML template of the default configuration."""
    out = Path(path)
    out.write_text(EvaluationConfig().to_yaml(), encoding="utf-8")
    return out
