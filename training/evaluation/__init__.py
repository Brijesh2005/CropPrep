"""CropFusion evaluation framework (Phase R5).

Provides the deep evaluation surface produced after training:

* :class:`MultimodalEvaluator` — runs a trained model over a loader and
  reduces per-task metrics, PR curves, confusion matrices, raw predictions,
  shared embeddings and forward latency into an :class:`EvaluationOutcome`.
* Extended metrics — balanced accuracy, per-class precision / recall / F1,
  ROC-AUC, AUPRC; median absolute error, bias, error percentiles, residual
  histogram and within-tolerance fraction.
* Comparison tables — per-class, regression error-profile and a multimodal
  comparison across models / variants.
* :class:`AblationStudy` — the seven R5 ablation variants with metric /
  parameter / inference-speed impact.
* :class:`ErrorAnalysis` — misclassifications, outliers, failure cases and
  group breakdowns.
* Report generators — markdown / JSON / CSV / PNG artefacts.
"""

from __future__ import annotations

from .ablation import (
    ABLATION_VARIANTS,
    DEFAULT_VARIANTS,
    AblationStudy,
    AblationStudyReport,
    apply_variant_surgery,
    build_variant_config,
)
from .comparison import (
    build_classification_comparison,
    build_multimodal_comparison,
    build_regression_comparison,
    render_comparison_markdown,
    render_markdown_table,
)
from .config import (
    AblationConfig,
    ComparisonConfig,
    ErrorAnalysisConfig,
    EvaluationConfig,
    GeneralConfig,
    MetricsConfig,
    load_evaluation_config,
    save_evaluation_template,
)
from .error_analysis import ErrorAnalysis, ErrorAnalysisReport
from .evaluator import EvaluationOutcome, MultimodalEvaluator
from .exceptions import (
    AblationStudyError,
    ComparisonError,
    ErrorAnalysisError,
    EvaluationConfigurationError,
    EvaluationError,
    EvaluationReportError,
    MetricComputationError,
)
from .metrics import (
    EvaluationAccumulator,
    compute_classification_metrics,
    compute_pr_curves,
    compute_regression_metrics,
)
from .reports import (
    generate_ablation_reports,
    generate_comparison_report,
    generate_error_analysis_reports,
    generate_evaluation_reports,
)

__version__ = "0.1.0"

__all__ = [
    # Config
    "EvaluationConfig",
    "GeneralConfig",
    "MetricsConfig",
    "ComparisonConfig",
    "AblationConfig",
    "ErrorAnalysisConfig",
    "load_evaluation_config",
    "save_evaluation_template",
    # Evaluator
    "MultimodalEvaluator",
    "EvaluationOutcome",
    # Metrics
    "EvaluationAccumulator",
    "compute_classification_metrics",
    "compute_regression_metrics",
    "compute_pr_curves",
    # Comparison
    "build_classification_comparison",
    "build_regression_comparison",
    "build_multimodal_comparison",
    "render_markdown_table",
    "render_comparison_markdown",
    # Ablation
    "AblationStudy",
    "AblationStudyReport",
    "ABLATION_VARIANTS",
    "DEFAULT_VARIANTS",
    "build_variant_config",
    "apply_variant_surgery",
    # Error analysis
    "ErrorAnalysis",
    "ErrorAnalysisReport",
    # Reports
    "generate_evaluation_reports",
    "generate_ablation_reports",
    "generate_error_analysis_reports",
    "generate_comparison_report",
    # Exceptions
    "EvaluationError",
    "EvaluationConfigurationError",
    "MetricComputationError",
    "ComparisonError",
    "AblationStudyError",
    "ErrorAnalysisError",
    "EvaluationReportError",
]
