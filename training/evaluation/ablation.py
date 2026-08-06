"""Ablation study over the CropFusion architecture (Phase R5).

Builds the seven R5 ablation variants — without TabTransformer, without
EfficientNet, without Temporal Encoder, without Cross Attention, without
Adaptive Gate, without Confidence Fusion, without Temporal Branch — from a base
:class:`ModelConfig`, then measures each one's task metrics, parameter count
and inference latency so the report can rank the architectural contribution of
every component.

Two mechanisms realise the variants:

* **config overrides** — the architecture already exposes additive toggles
  (``tabular`` dims, ``image_encoder.backbone``, ``cross_attention.enabled``,
  ``gated_fusion.enabled``, ``fusion.residual_fusion``,
  ``fusion.use_temporal_stream``), so most variants are pure configuration;
* **model surgery** — the temporal transformer has no config toggle, so
  "without temporal encoder" swaps in a mask-aware temporal-pooling module of
  identical input/output width (removing the transformer stack while keeping
  the temporal aggregation contract).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from shared.config import deep_merge
from training.models import ModelConfig, ModelFactory, count_parameters

from .comparison import build_multimodal_comparison
from .config import AblationConfig, EvaluationConfig
from .evaluator import EvaluationOutcome, MultimodalEvaluator
from .exceptions import AblationStudyError

#: Registry of the R5 ablation variants.
#: ``config_overrides`` are deep-merged into the base ``ModelConfig``;
#: ``surgery=True`` means the variant needs :func:`apply_variant_surgery`.
ABLATION_VARIANTS: list[dict[str, Any]] = [
    {
        "name": "without_tabtransformer",
        "description": (
            "Tabular branch (TabTransformer) removed — image only."
        ),
        "config_overrides": {
            "tabular": {"numeric_dim": 0, "categorical_cardinalities": []}
        },
        "surgery": False,
    },
    {
        "name": "without_efficientnet",
        "description": (
            "Image branch (EfficientNetV2 backbone) removed — tabular only."
        ),
        "config_overrides": {
            "image_encoder": {"backbone": None},
            "fusion": {"use_temporal_stream": False},
        },
        "surgery": False,
    },
    {
        "name": "without_temporal_encoder",
        "description": (
            "Temporal transformer replaced by mask-aware mean pooling "
            "(same input/output width)."
        ),
        "config_overrides": {},
        "surgery": True,
    },
    {
        "name": "without_cross_attention",
        "description": (
            "Cross-attention block removed; gated fusion still active."
        ),
        "config_overrides": {"cross_attention": {"enabled": False}},
        "surgery": False,
    },
    {
        "name": "without_adaptive_gate",
        "description": (
            "Adaptive gated fusion removed — image + tabular streams are "
            "concatenated into the shared encoder."
        ),
        "config_overrides": {"gated_fusion": {"enabled": False}},
        "surgery": False,
    },
    {
        "name": "without_confidence_fusion",
        "description": (
            "Confidence-weighted fusion removed — gating off AND the residual "
            "re-injection of the raw modality streams disabled."
        ),
        "config_overrides": {
            "gated_fusion": {"enabled": False},
            "fusion": {"residual_fusion": False},
        },
        "surgery": False,
    },
    {
        "name": "without_temporal_branch",
        "description": (
            "The temporal stream (fourth gate feeding the gated fusion) is "
            "disabled."
        ),
        "config_overrides": {"fusion": {"use_temporal_stream": False}},
        "surgery": False,
    },
]

DEFAULT_VARIANTS = tuple(
    variant["name"] for variant in ABLATION_VARIANTS
)


def build_variant_config(base_config: ModelConfig, variant: str) -> ModelConfig:
    """Build a :class:`ModelConfig` for one ablation variant."""
    spec = _variant_spec(variant)
    merged = deep_merge(base_config.model_dump(), spec["config_overrides"])
    return ModelConfig.model_validate(merged)


def apply_variant_surgery(model: nn.Module, variant: str) -> nn.Module:
    """Apply structural surgery to a freshly built model for ``variant``."""
    spec = _variant_spec(variant)
    if not spec["surgery"]:
        return model
    if variant != "without_temporal_encoder":
        raise AblationStudyError(f"no surgery defined for variant {variant!r}")

    temporal = getattr(model, "temporal_transformer", None)
    if temporal is None:
        raise AblationStudyError(
            "cannot apply 'without_temporal_encoder' surgery: model has no "
            "temporal transformer (image branch disabled)"
        )
    input_dim = int(getattr(temporal, "input_proj", None).in_features)
    output_dim = int(getattr(temporal, "output_dim", input_dim))
    model.temporal_transformer = _TemporalPooling(input_dim, output_dim)
    return model


class _TemporalPooling(nn.Module):
    """Mask-aware temporal pooling — the ``without_temporal_encoder`` stand-in.

    Aggregates the fused per-timestep features with a mask-aware mean (padded
    observations never contribute), then projects to the image-embedding width.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.output_dim = output_dim

    def forward(  # type: ignore[override]
        self,
        sequence: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mask is not None:
            weights = mask.float().unsqueeze(-1)  # [B, T, 1]
            denominator = weights.sum(dim=1).clamp_min(1e-6)
            pooled = (sequence * weights).sum(dim=1) / denominator
        else:
            pooled = sequence.mean(dim=1)
        return self.norm(self.input_proj(pooled))


def _variant_spec(variant: str) -> dict[str, Any]:
    for spec in ABLATION_VARIANTS:
        if spec["name"] == variant:
            return spec
    raise AblationStudyError(
        f"unknown ablation variant {variant!r}",
        detail=sorted(spec["name"] for spec in ABLATION_VARIANTS),
    )


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


@dataclass
class AblationStudyReport:
    """Results of an ablation sweep over one base configuration."""

    base_name: str
    compare_metric: str
    compare_mode: str
    results: dict[str, dict[str, Any]] = field(default_factory=dict)
    comparison: dict[str, Any] = field(default_factory=dict)
    best_variant: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_name": self.base_name,
            "compare_metric": self.compare_metric,
            "compare_mode": self.compare_mode,
            "best_variant": self.best_variant,
            "results": self.results,
            "comparison": self.comparison,
        }


class AblationStudy:
    """Run the seven R5 variants and compare metrics / size / speed.

    Args:
        base_model: A trained (or freshly built) full CropFusion model whose
            ``config`` defines the architecture to ablate.
        config: Validated :class:`EvaluationConfig` (``None`` = defaults).
        device: Override the configured device.
    """

    def __init__(
        self,
        base_model: nn.Module,
        config: EvaluationConfig | None = None,
        *,
        device: str | None = None,
    ) -> None:
        self.base_model = base_model
        self.config = config or EvaluationConfig()
        self.device = device or self.config.general.device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._base_config = getattr(base_model, "config", None)
        if self._base_config is None:
            raise AblationStudyError(
                "base model must expose a ModelConfig (CropFusionModel)"
            )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run(
        self,
        loader: Any,
        variants: Sequence[str] | None = None,
    ) -> AblationStudyReport:
        """Build and evaluate every requested variant over ``loader``."""
        names = list(variants) if variants is not None else list(DEFAULT_VARIANTS)
        for name in names:
            _variant_spec(name)

        ablation_cfg = self.config.ablation
        report = AblationStudyReport(
            base_name=self._base_config.name,
            compare_metric=ablation_cfg.compare_metric,
            compare_mode=ablation_cfg.compare_mode,
        )
        base_params = count_parameters(self.base_model)

        for name in names:
            model = self._build_variant(name)
            outcome = MultimodalEvaluator(
                model, self.config, device=self.device
            ).evaluate(loader)
            params = count_parameters(model)
            speed = self._benchmark(model, ablation_cfg)

            report.results[name] = {
                "metrics": {
                    task: dict(outcome.metrics[task]) for task in outcome.metrics
                },
                "parameter_count": params,
                "parameter_delta": params - base_params,
                "inference_ms": speed,
                "speedup_vs_full": (
                    None
                ),
            }
            # Speedup is computed after the loop once the full baseline speed
            # is known (the first variant may not be the full model).

        full_speed = self._reference_speed(loader)
        for name in names:
            data = report.results[name]
            data["speedup_vs_full"] = (
                float(full_speed / data["inference_ms"])
                if data["inference_ms"] and full_speed
                else None
            )

        comparison = build_multimodal_comparison(
            {
                name: self._outcome_from_result(report.results[name])
                for name in names
            }
        )
        report.comparison = comparison
        report.best_variant = self._best_variant(report)
        return report

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _build_variant(self, name: str) -> nn.Module:
        variant_config = build_variant_config(self._base_config, name)
        model = ModelFactory.create(variant_config)
        model = apply_variant_surgery(model, name)
        model.to(self.device)
        model.eval()
        return model

    def _reference_speed(self, loader: Any) -> float | None:
        """Latency (ms) of the base model as the ablation baseline."""
        if not hasattr(self.base_model, "sample_batch"):
            return None
        ablation_cfg = self.config.ablation
        try:
            sample = self.base_model.sample_batch(batch_size=2)
        except Exception:
            return None
        sample = {
            k: v.to(self.device)
            for k, v in sample.items()
            if isinstance(v, torch.Tensor)
        }
        self.base_model.eval()
        with torch.no_grad():
            for _ in range(ablation_cfg.benchmark_warmup):
                self.base_model(sample)
            start = torch.cuda.Event(enable_timing=True) if self.device.startswith("cuda") else None
            times: list[float] = []
            for _ in range(ablation_cfg.benchmark_iterations):
                if start is not None:
                    start.record()
                    self.base_model(sample)
                    torch.cuda.synchronize()
                    times.append(start.elapsed_time())
                else:
                    import time

                    t0 = time.perf_counter()
                    self.base_model(sample)
                    times.append((time.perf_counter() - t0) * 1000.0)
        return float(np.mean(times))

    def _benchmark(self, model: nn.Module, cfg: AblationConfig) -> float:
        if not hasattr(model, "sample_batch"):
            return 0.0
        try:
            sample = model.sample_batch(batch_size=2)
        except Exception:
            return 0.0
        sample = {
            k: v.to(self.device)
            for k, v in sample.items()
            if isinstance(v, torch.Tensor)
        }
        model.eval()
        with torch.no_grad():
            for _ in range(cfg.benchmark_warmup):
                model(sample)
            start = torch.cuda.Event(enable_timing=True) if self.device.startswith("cuda") else None
            times: list[float] = []
            for _ in range(cfg.benchmark_iterations):
                if start is not None:
                    start.record()
                    model(sample)
                    torch.cuda.synchronize()
                    times.append(start.elapsed_time())
                else:
                    import time

                    t0 = time.perf_counter()
                    model(sample)
                    times.append((time.perf_counter() - t0) * 1000.0)
        return float(np.mean(times))

    @staticmethod
    def _outcome_from_result(result: dict[str, Any]) -> EvaluationOutcome:
        outcome = EvaluationOutcome()
        outcome.metrics = result["metrics"]
        return outcome

    def _best_variant(self, report: AblationStudyReport) -> str | None:
        metric = report.compare_metric
        scored: list[tuple[float, str]] = []
        for name, data in report.results.items():
            value = _metric_lookup(data["metrics"], metric)
            if value is None:
                continue
            scored.append((float(value), name))
        if not scored:
            return None
        scored.sort(
            key=lambda pair: pair[0],
            reverse=report.compare_mode == "max",
        )
        return scored[0][1]


def _metric_lookup(metrics: Mapping[str, Any], key: str) -> Any:
    if "/" not in key:
        return None
    task, metric = key.split("/", 1)
    return metrics.get(task, {}).get(metric)
