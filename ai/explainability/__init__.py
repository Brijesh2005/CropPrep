"""CropFusion explainability framework (Phase 7).

A complete Multimodal eXplainable AI (MXAI) framework for the Phase 5 model:

* :class:`Explainer` — the public facade: ``explain`` / ``explain_crop`` /
  ``explain_yield`` / ``generate_report`` / ``visualize`` / ``export``.
* SHAP (self-contained KernelSHAP) for the TabTransformer + force / waterfall /
  decision / summary / bar / dependence / interaction plots.
* GradCAM++ / GradCAM / EigenCAM / LayerCAM image heatmaps for NDVI & EVI.
* Temporal attention (attention rollout + observation importance).
* Cross-modal attention (gates + cross-attention + token importance).
* Integrated gradients for tabular / image / shared embedding.
* Counterfactual ("what-if") explanations.
* Uncertainty estimation (confidence, entropy, MC-dropout, ECE).
* Unified farmer-friendly + research reports, visualization and export
  (HTML / JSON / PNG / CSV / PDF).

The framework consumes the Phase 4 sample dict — no direct file loading.
"""

from __future__ import annotations

from .config import (
    CamConfig,
    CounterfactualConfig,
    CrossModalConfig,
    ExplainabilityConfig,
    ExportConfig,
    GeneralConfig,
    IntegratedGradientsConfig,
    ReportConfig,
    ShapConfig,
    TemporalAttentionConfig,
    UncertaintyConfig,
    VisualizationConfig,
    load_explainability_config,
    save_explainability_template,
)
from .counterfactual import CounterfactualEngine
from .cross_modal_attention import CrossModalExplainer
from .exceptions import (
    AttentionError,
    AttributionError,
    CamError,
    CounterfactualError,
    ExplainabilityConfigurationError,
    ExplainabilityError,
    ExportError,
    ReportError,
    ShapError,
    UncertaintyError,
    VisualizationError,
)
from .exporter import Exporter
from .facade import Explainer
from .gradcam import (
    EigenCAM,
    GradCAM,
    GradCAMPlusPlus,
    ImageExplainer,
    LayerCAM,
    compute_cam,
)
from .integrated_gradients import (
    ImageIntegratedGradients,
    IntegratedGradients,
    SharedEmbeddingIntegratedGradients,
    TabularIntegratedGradients,
)
from .report_generator import Explanation, ReportGenerator
from .shap_explainer import SHAPExplainer, ShapResult
from .temporal_attention import TemporalAttentionExplainer
from .uncertainty import UncertaintyEstimator
from .utils import AttentionCapture
from .visualization import Visualizer

__version__ = "0.1.0"

__all__ = [
    # Config
    "ExplainabilityConfig",
    "GeneralConfig",
    "ShapConfig",
    "CamConfig",
    "TemporalAttentionConfig",
    "CrossModalConfig",
    "IntegratedGradientsConfig",
    "CounterfactualConfig",
    "UncertaintyConfig",
    "ReportConfig",
    "VisualizationConfig",
    "ExportConfig",
    "load_explainability_config",
    "save_explainability_template",
    # Public API
    "Explainer",
    "Explanation",
    # SHAP
    "SHAPExplainer",
    "ShapResult",
    # GradCAM
    "ImageExplainer",
    "GradCAM",
    "GradCAMPlusPlus",
    "EigenCAM",
    "LayerCAM",
    "compute_cam",
    # Integrated gradients
    "IntegratedGradients",
    "TabularIntegratedGradients",
    "ImageIntegratedGradients",
    "SharedEmbeddingIntegratedGradients",
    # Attention
    "TemporalAttentionExplainer",
    "CrossModalExplainer",
    "AttentionCapture",
    # Uncertainty / counterfactual
    "UncertaintyEstimator",
    "CounterfactualEngine",
    # Report / visualization / export
    "ReportGenerator",
    "Visualizer",
    "Exporter",
    # Exceptions
    "ExplainabilityError",
    "ExplainabilityConfigurationError",
    "ShapError",
    "CamError",
    "AttentionError",
    "AttributionError",
    "CounterfactualError",
    "UncertaintyError",
    "ExportError",
    "ReportError",
    "VisualizationError",
]
