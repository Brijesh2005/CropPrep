"""Public explainability facade.

:class:`Explainer` is the single entry point for the MXAI framework. Given a
trained model, a fitted Phase 4 preprocessor and an :class:`AgriculturalObservation`, it
produces a unified :class:`Explanation`:

    prediction -> explanation engine -> tabular + image + temporal +
    cross-modal + confidence -> unified report

and offers :meth:`generate_report`, :meth:`visualize` and :meth:`export`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .config import ExplainabilityConfig
from .exceptions import ExplainabilityError
from .exporter import Exporter
from .gradcam import ImageExplainer
from .integrated_gradients import (
    ImageIntegratedGradients,
    SharedEmbeddingIntegratedGradients,
    TabularIntegratedGradients,
)
from .report_generator import Explanation, ReportGenerator
from .shap_explainer import SHAPExplainer
from .uncertainty import UncertaintyEstimator
from .utils import (
    crop_classes as _crop_classes_fn,
    feature_names as _feature_names_fn,
    inverse_scale_yield,
    outputs_to_task,
    single_sample_batch,
    to_numpy,
)
from .counterfactual import CounterfactualEngine
from .cross_modal_attention import CrossModalExplainer
from .temporal_attention import TemporalAttentionExplainer
from .visualization import Visualizer


class Explainer:
    """Explain any CropFusion prediction.

    Args:
        model: The trained :class:`~ai.models.cropfusion.CropFusionModel`.
        preprocessor: Fitted Phase 4 :class:`~ai.preprocessing.Preprocessor`
            (provides feature names, crop classes, yield scaling and sample
            transforms).
        config: Validated :class:`ExplainabilityConfig`.
        observations: Accepted observations used for SHAP background and the
            historical comparison.
        extractor: Patch extractor (e.g. ``STAM.get_patch``) for transforming
            observations with imagery.
        feature_names / crop_classes: Explicit overrides (else derived from the
            preprocessor).
    """

    def __init__(
        self,
        model: nn.Module,
        preprocessor: Any | None = None,
        config: ExplainabilityConfig | None = None,
        observations: Sequence[Any] | None = None,
        extractor: Any | None = None,
        feature_names: Sequence[str] | None = None,
        crop_classes: Sequence[str] | None = None,
    ) -> None:
        self.model = model
        self.preprocessor = preprocessor
        self.config = config or ExplainabilityConfig()
        self.observations = list(observations or [])
        self.extractor = extractor
        self.device = torch.device(
            "cuda"
            if self.config.general.device == "auto" and torch.cuda.is_available()
            else "cpu"
        )
        if feature_names is not None:
            self._feature_names = list(feature_names)
        elif preprocessor is not None:
            self._feature_names = _feature_names_fn(preprocessor)
        else:
            self._feature_names = []
        if crop_classes is not None:
            self._crop_classes = list(crop_classes)
        elif preprocessor is not None:
            self._crop_classes = _crop_classes_fn(preprocessor)
        else:
            self._crop_classes = []

        self._explainers: dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # Lazily built explainers
    # ------------------------------------------------------------------ #

    def _shap(self) -> SHAPExplainer:
        if "shap" not in self._explainers:
            self._explainers["shap"] = SHAPExplainer(
                self.model, self.config.shap, self.device
            )
        return self._explainers["shap"]

    def _uncertainty(self) -> UncertaintyEstimator:
        if "uncertainty" not in self._explainers:
            self._explainers["uncertainty"] = UncertaintyEstimator(
                self.model, self.config.uncertainty, self.device
            )
        return self._explainers["uncertainty"]

    def _counterfactual(self) -> CounterfactualEngine:
        if "counterfactual" not in self._explainers:
            self._explainers["counterfactual"] = CounterfactualEngine(
                self.model, self.config.counterfactual, self.device, self._feature_names
            )
        return self._explainers["counterfactual"]

    # ------------------------------------------------------------------ #
    # Sample / background
    # ------------------------------------------------------------------ #

    def sample(self, observation: Any) -> dict[str, torch.Tensor]:
        """Transform an observation into an AI-ready sample (Phase 4)."""
        if self.preprocessor is None:
            raise ExplainabilityError("a fitted preprocessor is required")
        if not getattr(self.preprocessor, "fitted", False):
            raise ExplainabilityError("preprocessor must be fitted before explanation")
        return self.preprocessor.transform(observation, extractor=self.extractor)

    def background(self, target_sample: Mapping[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        """Reference samples for SHAP: background tabulars + the target images.

        Keeping the target's imagery fixed isolates the tabular feature
        attribution that SHAP is explaining.
        """
        size = self.config.shap.background_size
        size = max(1, min(size, len(self.observations)))
        if size == 0:
            raise ExplainabilityError("no background observations provided")
        rng = np.random.RandomState(self.config.general.seed)
        indices = rng.choice(len(self.observations), size=size, replace=False)
        tabulars: list[torch.Tensor] = []
        for idx in indices:
            obs = self.observations[int(idx)]
            try:
                vec = self.preprocessor.tabular.transform(obs)
            except Exception:
                continue
            tabulars.append(vec)
        if not tabulars:
            raise ExplainabilityError("could not build a tabular background")

        background: list[dict[str, torch.Tensor]] = []
        for vec in tabulars:
            item: dict[str, torch.Tensor] = {"tabular": vec}
            for key in ("ndvi", "evi", "temporal_mask"):
                if key in target_sample:
                    item[key] = target_sample[key]
            background.append(item)
        return background

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def predict(self, observation: Any) -> dict[str, Any]:
        """Raw prediction (logits / scaled yield) for an observation."""
        sample = self.sample(observation)
        out = self.model(single_sample_batch(sample, self.device))
        result: dict[str, Any] = {"sample": sample}
        tasks = outputs_to_task(out)
        if "crop" in tasks:
            result["crop_logits"] = to_numpy(tasks["crop"])[0]
            result["crop_class"] = int(result["crop_logits"].argmax())
            result["crop_probs"] = to_numpy(
                torch.softmax(torch.as_tensor(result["crop_logits"]).float(), dim=-1)
            )
        if "yield" in tasks:
            result["yield_scaled"] = float(tasks["yield"][0, 0].item())
            result["yield_prediction"] = inverse_scale_yield(
                self.preprocessor, result["yield_scaled"]
            )
        result["gates"] = {
            k: float(v.reshape(-1)[0].item())
            for k, v in (getattr(out, "gates", {}) or {}).items()
        }
        return result

    def explain(self, observation: Any, *, task: str = "crop") -> Explanation:
        """Full multimodal explanation for one observation."""
        sample = self.sample(observation)
        prediction = self.predict(observation)
        classes = self._crop_classes or [f"class_{i}" for i in range(max(int(prediction.get("crop_class", 0) or 0) + 1, 1))]

        crop_class = int(prediction.get("crop_class", 0) or 0)
        crop_name = classes[crop_class] if crop_class < len(classes) else str(crop_class)
        target_kind = task if task in ("crop", "yield") else "crop"

        explanation = Explanation(
            observation_id=str(getattr(observation, "observation_id", "")),
            crop=crop_name,
            crop_probs=(
                {classes[i]: float(p) for i, p in enumerate(prediction.get("crop_probs", []))}
                if prediction.get("crop_probs") is not None
                else {}
            ),
            yield_prediction=prediction.get("yield_prediction"),
            gates=prediction.get("gates", {}),
            raw={"sample": {k: v.clone() if torch.is_tensor(v) else v for k, v in sample.items()}},
        )

        # -- Confidence / uncertainty ------------------------------------ #
        uncertainty = self._uncertainty()
        batch = single_sample_batch(sample, self.device)
        confidence: dict[str, Any] = {}
        if prediction.get("crop_probs") is not None:
            confidence["crop_conf"] = float(np.max(prediction["crop_probs"]))
            confidence["crop_entropy"] = float(
                -np.sum(prediction["crop_probs"] * np.log(np.clip(prediction["crop_probs"], 1e-12, 1.0)))
            )
            confidence["predicted_class"] = crop_class
        if self.config.uncertainty.mc_dropout_samples > 0:
            try:
                mc = uncertainty.mc_dropout(batch, samples=self.config.uncertainty.mc_dropout_samples)
                confidence.update({k: v for k, v in mc.items() if k not in ("crop_probs",)})
            except Exception:
                pass
        explanation.confidence = confidence

        # -- SHAP (tabular) ---------------------------------------------- #
        if "tabular" in sample:
            try:
                background = self.background(sample)
                shap_result = self._shap().explain(
                    sample, background, kind=target_kind, target_class=crop_class,
                    feature_names_=self._feature_names,
                )
                explanation.feature_importance = {
                    name: float(value) for name, value in zip(self._feature_names, shap_result.values)
                }
                explanation.shap_values = shap_result.values
                explanation.shap_base_value = shap_result.base_value
            except Exception as exc:
                explanation.limitations.append(f"SHAP unavailable: {exc}")

        # -- Integrated gradients ---------------------------------------- #
        explanation.integrated_gradients = self._integrated_gradients(
            sample, target_kind, crop_class
        )

        # -- Image / temporal / cross-modal ------------------------------ #
        if getattr(self.model, "use_image", False) and "ndvi" in sample:
            self._image_explanations(observation, sample, explanation, target_kind, crop_class)

        # -- Counterfactuals --------------------------------------------- #
        try:
            cf = self._counterfactual().explain(sample)
            explanation.counterfactuals = cf.get("counterfactuals", [])
        except Exception as exc:
            explanation.limitations.append(f"Counterfactuals unavailable: {exc}")

        # -- Historical + reasoning -------------------------------------- #
        report_generator = ReportGenerator(
            self.config.report, self.observations, self._crop_classes
        )
        explanation.historical = report_generator.historical_comparison(explanation)
        explanation.reasoning = report_generator.build_reasoning(explanation)
        explanation.limitations.extend(report_generator.build_limitations(explanation))
        return explanation

    def explain_crop(self, observation: Any) -> Explanation:
        return self.explain(observation, task="crop")

    def explain_yield(self, observation: Any) -> Explanation:
        return self.explain(observation, task="yield")

    # ------------------------------------------------------------------ #
    # Report / visualize / export
    # ------------------------------------------------------------------ #

    def generate_report(
        self, observation: Any, mode: str = "farmer"
    ) -> dict[str, Any]:
        explanation = self.explain(observation)
        generator = ReportGenerator(self.config.report, self.observations, self._crop_classes)
        return (
            generator.farmer_report(explanation)
            if mode == "farmer"
            else generator.research_report(explanation)
        )

    def visualize(
        self,
        explanation: Explanation,
        output_dir: str | Path | None = None,
    ) -> dict[str, Path]:
        """Render all explanation figures and return ``{name: path}``."""
        visualizer = Visualizer(
            output_dir or self.config.visualization.directory,
            dpi=self.config.visualization.dpi,
            colormap=self.config.visualization.colormap,
            max_features=self.config.visualization.max_features_bar,
        )
        artifacts: dict[str, Path] = {}

        if explanation.shap_values is not None:
            names = self._feature_names
            artifacts["feature_importance"] = visualizer.feature_importance_bar(
                explanation.shap_values, names, visualizer.output_dir / "feature_importance.png"
            )
            if len(names) <= self.config.visualization.max_features_bar:
                artifacts["shap_waterfall"] = visualizer.shap_waterfall(
                    explanation.shap_values, explanation.shap_base_value or 0.0, names,
                    visualizer.output_dir / "shap_waterfall.png",
                )
                artifacts["shap_force"] = visualizer.shap_force(
                    explanation.shap_values, explanation.shap_base_value or 0.0, names,
                    visualizer.output_dir / "shap_force.png",
                )
                artifacts["shap_decision"] = visualizer.shap_decision(
                    explanation.shap_values, explanation.shap_base_value or 0.0, names,
                    visualizer.output_dir / "shap_decision.png",
                )

        for index in ("ndvi", "evi"):
            overlay = explanation.image_overlays.get(index)
            if overlay is not None:
                artifacts[f"gradcam_{index}"] = visualizer.gradcam_overlay(
                    overlay, visualizer.output_dir / f"gradcam_{index}.png"
                )

        temporal = explanation.temporal_importance
        if temporal:
            artifacts["temporal_timeline"] = visualizer.temporal_timeline(
                np.asarray(list(temporal.values())),
                list(temporal.keys()),
                visualizer.output_dir / "temporal_timeline.png",
            )

        cross = explanation.cross_modal.get("cross_modal_heatmap")
        if cross is not None:
            artifacts["cross_modal_heatmap"] = visualizer.cross_modal_heatmap(
                np.asarray(cross),
                list(explanation.temporal_importance.keys()),
                self._feature_names,
                visualizer.output_dir / "cross_modal_heatmap.png",
            )

        if explanation.confidence.get("crop_conf") is not None:
            artifacts["confidence"] = visualizer.confidence_distribution(
                np.asarray([explanation.confidence["crop_conf"]]),
                path=visualizer.output_dir / "confidence.png",
            )
        return artifacts

    def export(
        self,
        explanation: Explanation,
        *,
        output_dir: str | Path | None = None,
        formats: list[str] | None = None,
        figures: Mapping[str, Path] | None = None,
    ) -> dict[str, Path]:
        """Export the explanation to HTML / JSON / PNG / CSV / PDF."""
        return Exporter(self.config.export, output_dir).export(
            explanation, formats=formats, figures=figures
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _integrated_gradients(
        self, sample: Mapping[str, torch.Tensor], kind: str, class_index: int
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if "tabular" in sample:
            try:
                ig = TabularIntegratedGradients(self.model, self.config.integrated_gradients, self.device)
                values = ig.attribute(sample, kind=kind, class_index=class_index)
                result["tabular"] = {
                    name: float(v)
                    for name, v in zip(self._feature_names, np.asarray(values).reshape(-1))
                }
            except Exception as exc:
                result["tabular_error"] = str(exc)
        try:
            ig_shared = SharedEmbeddingIntegratedGradients(
                self.model, self.config.integrated_gradients, self.device
            )
            result["shared_embedding"] = np.asarray(
                ig_shared.attribute(sample, kind=kind, class_index=class_index)
            ).tolist()
        except Exception as exc:
            result["shared_error"] = str(exc)
        return result

    def _image_explanations(
        self,
        observation: Any,
        sample: Mapping[str, torch.Tensor],
        explanation: Explanation,
        kind: str,
        class_index: int,
    ) -> None:
        # GradCAM heatmaps + overlays.
        image_explainer = ImageExplainer(self.model, self.config.cam, self.device)
        for index in ("ndvi", "evi"):
            if index not in sample:
                continue
            try:
                result = image_explainer.explain(
                    sample, index=index, kind=kind, class_index=class_index
                )
                explanation.image_heatmaps[index] = result
                explanation.image_overlays[index] = image_explainer.overlay(
                    sample, result, index=index
                )
            except Exception as exc:
                explanation.limitations.append(f"{index.upper()} GradCAM unavailable: {exc}")

        # Temporal importance.
        try:
            temporal = TemporalAttentionExplainer(
                self.model, self.config.temporal_attention, self.device
            ).explain(sample)
            dates = _observation_dates(observation, int(temporal.get("timesteps", 0)))
            importance = temporal["importance"]
            explanation.temporal_importance = {
                dates[i] if i < len(dates) else f"obs_{i}": float(importance[i])
                for i in range(int(temporal.get("timesteps", 0)))
            }
            explanation.temporal_ranking = [
                dates[i] if i < len(dates) else f"obs_{i}"
                for i in temporal.get("ranking", [])
            ]
            explanation.raw["temporal"] = temporal
        except Exception as exc:
            explanation.limitations.append(f"Temporal attention unavailable: {exc}")

        # Cross-modal.
        try:
            cross = CrossModalExplainer(
                self.model, self.config.cross_modal, self.device
            ).explain(sample, feature_names=self._feature_names)
            explanation.cross_modal = {
                k: (v if k != "cross_modal_heatmap" else v)
                for k, v in cross.items()
            }
        except Exception as exc:
            explanation.limitations.append(f"Cross-modal explanation unavailable: {exc}")


def _observation_dates(observation: Any, timesteps: int) -> list[str]:
    pairs = getattr(getattr(observation, "sequence", None), "pairs", None) or []
    dates = sorted(
        [pair.date.isoformat() for pair in pairs if getattr(pair, "date", None)]
    )
    dates = dates[:timesteps]
    while len(dates) < timesteps:
        dates.append("")
    return dates
