"""Ablation experiments.

The Phase 5 model was extended with additive, backward-compatible toggles
(``enable_ndvi`` / ``enable_evi`` / ``cross_attention.enabled`` /
``gated_fusion.enabled``) so the required ablations can be expressed purely as
configuration changes. :class:`AblationRunner` runs the same experiment under
each variant and produces an automatic comparison report.

Variants:

* ``full`` — the complete architecture,
* ``only_tabular`` — no image branch,
* ``only_ndvi`` — image branch uses the NDVI stream only,
* ``only_evi`` — image branch uses the EVI stream only,
* ``only_image`` — no tabular branch,
* ``no_cross_attention`` — cross-attention removed (gate still active),
* ``no_adaptive_gate`` — adaptive gated fusion removed (concat instead).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from training.models import ModelConfig
from training.dataset_manager.config import deep_merge

from .config import TrainingConfig
from .exceptions import AblationError

#: Configuration overrides that define each ablation variant.
ABLATION_VARIANTS: dict[str, Mapping[str, Any]] = {
    "full": {},
    "only_tabular": {"image_encoder": {"backbone": None}},
    "only_ndvi": {"image_encoder": {"enable_evi": False}},
    "only_evi": {"image_encoder": {"enable_ndvi": False}},
    "only_image": {
        "tabular": {"numeric_dim": 0, "categorical_cardinalities": []}
    },
    "no_cross_attention": {"cross_attention": {"enabled": False}},
    "no_adaptive_gate": {"gated_fusion": {"enabled": False}},
}

DEFAULT_VARIANTS = tuple(ABLATION_VARIANTS)


def build_variant_config(base_config: ModelConfig, variant: str) -> ModelConfig:
    """Build a :class:`ModelConfig` for one ablation variant."""
    if variant not in ABLATION_VARIANTS:
        raise AblationError(f"unknown ablation variant {variant!r}")
    merged = deep_merge(base_config.model_dump(), ABLATION_VARIANTS[variant])
    return ModelConfig.model_validate(merged)


@dataclass
class AblationReport:
    """Comparison of ablation variants for one base configuration."""

    base_name: str
    compare_metric: str
    compare_mode: str
    results: dict[str, dict[str, Any]] = field(default_factory=dict)
    best_variant: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_name": self.base_name,
            "compare_metric": self.compare_metric,
            "compare_mode": self.compare_mode,
            "best_variant": self.best_variant,
            "results": self.results,
        }


class AblationRunner:
    """Run the same experiment under every ablation variant and compare.

    Args:
        training_config: Validated :class:`TrainingConfig` (its
            ``ablation`` / ``validation`` sections drive the sweep).
        base_model_config: The full-model configuration (typically derived
            from the fitted preprocessor).
        preprocessor: Fitted Phase 4 :class:`Preprocessor`.
        observations: Accepted observations used for the hold-out split.
        extractor: Patch extractor (e.g. ``STAM.get_patch``).
        output_dir: Directory for the comparison report (defaults to
            ``<training output>/ablation``).
    """

    def __init__(
        self,
        training_config: TrainingConfig,
        base_model_config: ModelConfig,
        *,
        preprocessor: Any,
        observations: Sequence[Any],
        extractor: Any | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        self.config = training_config
        self.base_model_config = base_model_config
        self.preprocessor = preprocessor
        self.observations = list(observations)
        self.extractor = extractor
        self.output_dir = Path(
            output_dir or (training_config.general.output_dir / "ablation")
        )

    def run(self, variants: Sequence[str] | None = None) -> AblationReport:
        """Run every variant (or the requested subset) and compare results."""
        from .experiment import Experiment

        names = list(variants) if variants is not None else list(DEFAULT_VARIANTS)
        for name in names:
            if name not in ABLATION_VARIANTS:
                raise AblationError(f"unknown ablation variant {name!r}")

        cfg = self.config.ablation
        report = AblationReport(
            base_name=self.base_model_config.name,
            compare_metric=cfg.compare_metric,
            compare_mode=cfg.compare_mode,
        )

        for name in names:
            variant_config = build_variant_config(self.base_model_config, name)
            variant_dir = self.output_dir / name
            variant_dir.mkdir(parents=True, exist_ok=True)
            exp_config = self._variant_training_config(name)

            experiment = Experiment(
                exp_config,
                self.observations,
                preprocessor=self.preprocessor,
                extractor=self.extractor,
                model_config=variant_config,
                run_dir=variant_dir,
                run_name=name,
            )
            exp_report = experiment.run()

            metrics = self._extract_metrics(exp_report)
            report.results[name] = {
                "metrics": metrics,
                "run_dir": str(variant_dir),
                "multi_task_score": (
                    exp_report.evaluation.multi_task_score
                    if exp_report.evaluation is not None
                    else None
                ),
            }

        report.best_variant = self._best_variant(report)
        self._write_report(report)
        return report

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _variant_training_config(self, variant: str) -> TrainingConfig:
        """A copy of the training config with a per-variant run name."""
        import copy

        exp_config = copy.deepcopy(self.config)
        exp_config.name = f"{self.config.name}_{variant}"
        # Ablation runs use hold-out so results are directly comparable.
        if exp_config.validation.strategy != "holdout":
            exp_config.validation.strategy = "holdout"
        return exp_config

    def _extract_metrics(self, exp_report: Any) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        evaluation = exp_report.evaluation
        if evaluation is not None:
            for key, value in evaluation.metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    metrics[key] = value
            metrics["multi_task_score"] = evaluation.multi_task_score
        return metrics

    def _best_variant(self, report: AblationReport) -> str | None:
        scored: list[tuple[float, str]] = []
        for name, data in report.results.items():
            metric = data.get("metrics", {}).get(report.compare_metric)
            if metric is None:
                continue
            scored.append((float(metric), name))
        if not scored:
            return None
        scored.sort(
            key=lambda pair: pair[0],
            reverse=report.compare_mode == "max",
        )
        return scored[0][1]

    def _write_report(self, report: AblationReport) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "ablation_report.json"
        path.write_text(
            json.dumps(report.to_dict(), indent=2), encoding="utf-8"
        )
        # Human-friendly comparison table.
        rows = ["variant,multi_task_score"]
        for name, data in report.results.items():
            score = data.get("multi_task_score")
            rows.append(f"{name},{score if score is not None else ''}")
        (self.output_dir / "ablation_comparison.csv").write_text(
            "\n".join(rows) + "\n", encoding="utf-8"
        )
        return path
