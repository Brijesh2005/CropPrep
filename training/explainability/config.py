"""Configuration for the explainability framework.

Everything is configurable through YAML (or ``MXAI_*`` env vars), mirroring the
resolution order of the other CropFusion packages:

    env (``MXAI_<SECTION>__<KEY>``) > YAML (``MXAI_CONFIG_FILE``) > defaults

Every field is validated by pydantic. The root :class:`ExplainabilityConfig`
contains one section per explainer (SHAP, GradCAM, temporal attention,
cross-modal attention, integrated gradients, counterfactual, uncertainty) plus
report / visualization / export settings.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field

from training.dataset_manager.config import _apply_case_insensitive, _parse_env, deep_merge

from .exceptions import ExplainabilityConfigurationError

ENV_PREFIX = "MXAI_"


class GeneralConfig(BaseModel):
    """Device / seed / RNG settings."""

    model_config = ConfigDict(extra="forbid")

    device: str = "auto"  # auto | cpu | cuda (graceful CPU fallback)
    seed: int = 42


class ShapConfig(BaseModel):
    """SHAP attribution settings (self-contained KernelSHAP)."""

    model_config = ConfigDict(extra="forbid")

    #: kernel | gradient. ``kernel`` is the self-contained KernelSHAP;
    #: ``gradient`` uses the model gradient (Gradient x Input).
    method: str = Field(default="kernel", pattern="^(kernel|gradient)$")
    #: Background samples used as the reference distribution.
    background_size: int = Field(default=50, ge=1)
    #: Number of coalitions sampled per explanation (capped at 2^F).
    max_samples: int = Field(default=256, ge=8)
    #: Use the ``shap`` library when it is installed (falls back otherwise).
    prefer_library: bool = False
    #: Plots to produce (feature_importance, summary, waterfall, force,
    #: decision, bar, dependence, interaction).
    plots: list[str] = Field(
        default_factory=lambda: [
            "feature_importance",
            "summary",
            "waterfall",
            "bar",
        ]
    )


class CamConfig(BaseModel):
    """GradCAM family settings."""

    model_config = ConfigDict(extra="forbid")

    #: gradcam | gradcam++ | eigencam | layercam.
    method: str = Field(default="gradcam++", pattern="^(gradcam|gradcam\\+\\+|eigencam|layercam)$")
    #: Target layer regex override; ``null`` = last spatial conv of the backbone.
    target_layer: str | None = None
    #: ReLU the CAM (keep only positive evidence).
    relu: bool = True
    #: Colormap used for heatmaps.
    colormap: str = "jet"


class TemporalAttentionConfig(BaseModel):
    """Temporal attention / attention rollout settings."""

    model_config = ConfigDict(extra="forbid")

    #: Apply attention rollout (Abnar & Zuidema 2020) across layers.
    rollout: bool = True
    #: Include residual connections in the rollout.
    include_residual: bool = True
    #: Head aggregation: mean | max.
    head_aggregation: str = Field(default="mean", pattern="^(mean|max)$")
    #: Max timesteps to report (``None`` = all).
    max_timesteps: int | None = None


class CrossModalConfig(BaseModel):
    """Cross-modal attention settings."""

    model_config = ConfigDict(extra="forbid")

    #: Number of top tabular tokens to highlight.
    top_tokens: int = Field(default=5, ge=1)
    #: Row-normalise the cross-modal contribution heatmap.
    normalize: bool = True


class IntegratedGradientsConfig(BaseModel):
    """Integrated gradients settings."""

    model_config = ConfigDict(extra="forbid")

    steps: int = Field(default=50, ge=2)
    #: Baseline: zero | mean | random.
    baseline: str = Field(default="zero", pattern="^(zero|mean|random)$")
    #: Fraction of the sample used for a random baseline.
    random_fraction: float = Field(default=0.1, gt=0.0, le=1.0)


class CounterfactualConfig(BaseModel):
    """Counterfactual ("what-if") settings."""

    model_config = ConfigDict(extra="forbid")

    #: Default perturbations: feature -> {"delta": value, "mode": add|multiply|set}.
    perturbations: dict[str, dict[str, Any]] = Field(default_factory=dict)
    #: Max counterfactual examples to evaluate.
    max_examples: int = Field(default=10, ge=1)
    #: Threshold (fraction) above which a crop change counts as a switch.
    switch_threshold: float = Field(default=0.05, gt=0.0, le=1.0)


class UncertaintyConfig(BaseModel):
    """Confidence / uncertainty settings."""

    model_config = ConfigDict(extra="forbid")

    #: Monte-Carlo dropout samples (0 disables MC dropout).
    mc_dropout_samples: int = Field(default=10, ge=0)
    #: ECE / reliability-diagram bin count.
    bins: int = Field(default=10, ge=2)
    #: Confidence to report for yield (1 - normalized residual under MC).
    yield_confidence: str = Field(default="mc", pattern="^(mc|heuristic)$")


class ReportConfig(BaseModel):
    """Explanation report settings."""

    model_config = ConfigDict(extra="forbid")

    #: Number of top tabular features in the report.
    top_k_features: int = Field(default=10, ge=1)
    #: Number of top image regions reported.
    top_image_regions: int = Field(default=5, ge=1)
    #: Include historical comparison against training observations.
    include_historical: bool = True
    #: Farmer-friendly reasoning statements.
    farmer_reasoning: bool = True
    #: List limitations explicitly.
    include_limitations: bool = True


class VisualizationConfig(BaseModel):
    """Figure output settings."""

    model_config = ConfigDict(extra="forbid")

    directory: str = "artifacts/explainability/figures"
    dpi: int = Field(default=110, ge=50)
    colormap: str = "jet"
    max_features_bar: int = Field(default=15, ge=1)


class ExportConfig(BaseModel):
    """Explanation export settings."""

    model_config = ConfigDict(extra="forbid")

    directory: str = "artifacts/explainability/exports"
    formats: list[str] = Field(
        default_factory=lambda: ["html", "json", "png", "csv"]
    )
    pdf: bool = True


class ExplainabilityConfig(BaseModel):
    """Root explainability configuration."""

    model_config = ConfigDict(extra="forbid")

    name: str = "cropfusion_explainability"
    general: GeneralConfig = Field(default_factory=GeneralConfig)
    shap: ShapConfig = Field(default_factory=ShapConfig)
    cam: CamConfig = Field(default_factory=CamConfig)
    temporal_attention: TemporalAttentionConfig = Field(default_factory=TemporalAttentionConfig)
    cross_modal: CrossModalConfig = Field(default_factory=CrossModalConfig)
    integrated_gradients: IntegratedGradientsConfig = Field(default_factory=IntegratedGradientsConfig)
    counterfactual: CounterfactualConfig = Field(default_factory=CounterfactualConfig)
    uncertainty: UncertaintyConfig = Field(default_factory=UncertaintyConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)

    def to_yaml(self) -> str:
        return yaml.safe_dump(_yaml_safe(self.model_dump()), sort_keys=False)

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.write_text(self.to_yaml(), encoding="utf-8")
        return out


def load_explainability_config(
    config_path: str | Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> ExplainabilityConfig:
    """Load and validate explainability settings (env > YAML > defaults)."""
    env_map = dict(os.environ if env is None else env)
    if config_path is None:
        env_config = env_map.get("MXAI_CONFIG_FILE")
        config_path = env_config or None

    data: dict[str, Any] = {}
    if config_path is not None:
        config_file = Path(config_path)
        if not config_file.exists():
            raise ExplainabilityConfigurationError(
                f"Explainability config file not found: {config_file}",
                detail=str(config_file),
            )
        try:
            raw = yaml.safe_load(config_file.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ExplainabilityConfigurationError(
                f"Malformed explainability YAML: {exc}", detail=str(config_file)
            ) from exc
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ExplainabilityConfigurationError(
                "Explainability config root must be a mapping"
            )
        data = raw

    parsed_env = _parse_env(env_map, prefix=ENV_PREFIX)
    parsed_env.pop("config_file", None)
    merged = deep_merge(data, parsed_env)
    merged = _apply_case_insensitive(merged, ExplainabilityConfig)
    try:
        return ExplainabilityConfig.model_validate(merged)
    except Exception as exc:  # pydantic.ValidationError
        raise ExplainabilityConfigurationError(
            f"Invalid explainability configuration: {exc}"
        ) from exc


def save_explainability_template(path: str | Path) -> Path:
    """Write an annotated YAML template of the default configuration."""
    out = Path(path)
    out.write_text(ExplainabilityConfig().to_yaml(), encoding="utf-8")
    return out


def _yaml_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _yaml_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_yaml_safe(v) for v in value]
    return value
