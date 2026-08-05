"""CropFusion training & evaluation framework (Phase 6).

Implements the complete training engine consumed by the Phase 5 multimodal
model:

* :class:`Trainer` — AMP, gradient clipping / accumulation / checkpointing,
  NaN detection, early stopping, automatic resume, DDP with CPU fallback.
* :class:`Validator` — validation loop + hold-out / K-fold / stratified /
  spatial / temporal model validation.
* :class:`Evaluator` — final metrics, inference latency, memory, parameter
  count and the combined multi-task score.
* :class:`TrainingCheckpointManager` — best / latest / periodic checkpoints
  with optimizer, scheduler, AMP scaler and random state.
* Losses, optimizers (AdamW / SGD / RAdam / Lion), LR schedulers (cosine /
  OneCycle / ReduceLROnPlateau / polynomial / warmup).
* :class:`Experiment` — end-to-end orchestration; :class:`AblationRunner` —
  automatic ablation comparison.
* :class:`Benchmark` and :class:`Visualizer` — throughput/resource reports and
  automatic charts + dashboard.

The framework consumes exactly the Phase 4 batch dict (Dataset Manager → STAM
→ preprocessing → PyTorch DataLoader → model); no direct file loading.
"""

from __future__ import annotations

from .ablation import (
    ABLATION_VARIANTS,
    DEFAULT_VARIANTS,
    AblationReport,
    AblationRunner,
    build_variant_config,
)
from .benchmark import Benchmark, BenchmarkReport
from .callbacks import (
    ConsoleLogger,
    EarlyStopping,
    EarlyStopOnNan,
    HistoryRecorder,
    LearningRateLogger,
    ModelCheckpoint,
    TensorBoardCallback,
    WandbCallback,
)
from .checkpoint import (
    TrainingCheckpointManager,
    TrainingResumeState,
    capture_rng_state,
    restore_rng_state,
)
from .config import (
    AblationConfig,
    BenchmarkConfig,
    CheckpointConfig,
    DataConfig,
    GeneralConfig,
    LoggingConfig,
    LossConfig,
    MetricsConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainConfig,
    TrainingConfig,
    ValidationConfig,
    VisualizationConfig,
    load_training_config,
    save_training_template,
)
from .evaluator import EvaluationResult, Evaluator
from .exceptions import (
    AblationError,
    BenchmarkError,
    CheckpointError,
    LossBuildError,
    MetricError,
    OptimizerBuildError,
    SchedulerBuildError,
    TrainingConfigurationError,
    TrainingError,
    TrainingRunError,
    ValidationError,
    VisualizationError,
)
from .experiment import Experiment, ExperimentReport, run_experiment
from .interfaces import Callback, FoldGenerator, SchedulerHandle, Weighter
from .logger import ExperimentLogger
from .losses import (
    GradNormController,
    MAELoss,
    MultiTaskLoss,
    build_multi_task_loss,
    build_task_loss,
)
from .metrics import (
    ClassificationAccumulator,
    MetricsTracker,
    RegressionAccumulator,
    compute_classification_metrics,
    compute_regression_metrics,
)
from .optimizers import Lion, build_optimizer
from .schedulers import build_scheduler, get_lr
from .trainer import Trainer, TrainingResult
from .utils import (
    apply_gradient_checkpointing,
    cleanup_distributed,
    compute_grad_norm,
    configure_determinism,
    get_environment_info,
    get_git_branch,
    get_git_hash,
    is_distributed,
    is_primary,
    resolve_device,
    set_seed,
    setup_distributed,
    to_device,
)
from .validator import (
    FoldGenerator as _FoldGenerator,  # noqa: F401 (alias kept for imports)
)
from .validator import (
    HoldOutFoldGenerator,
    KFoldFoldGenerator,
    SpatialFoldGenerator,
    StratifiedKFoldFoldGenerator,
    TemporalFoldGenerator,
    Validator,
    ValidationResult,
    build_fold_generator,
    cross_validation_splits,
)
from .visualizer import Visualizer

__version__ = "0.1.0"

__all__ = [
    # Config
    "TrainingConfig",
    "GeneralConfig",
    "DataConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "LossConfig",
    "TrainConfig",
    "CheckpointConfig",
    "MetricsConfig",
    "LoggingConfig",
    "ValidationConfig",
    "AblationConfig",
    "BenchmarkConfig",
    "VisualizationConfig",
    "load_training_config",
    "save_training_template",
    # Engine
    "Trainer",
    "TrainingResult",
    "Validator",
    "ValidationResult",
    "Evaluator",
    "EvaluationResult",
    "Benchmark",
    "BenchmarkReport",
    # Losses / optimizers / schedulers
    "MultiTaskLoss",
    "MAELoss",
    "GradNormController",
    "build_multi_task_loss",
    "build_task_loss",
    "Lion",
    "build_optimizer",
    "build_scheduler",
    "get_lr",
    # Metrics
    "MetricsTracker",
    "ClassificationAccumulator",
    "RegressionAccumulator",
    "compute_classification_metrics",
    "compute_regression_metrics",
    # Checkpointing
    "TrainingCheckpointManager",
    "TrainingResumeState",
    "capture_rng_state",
    "restore_rng_state",
    # Callbacks / logging
    "Callback",
    "EarlyStopping",
    "ModelCheckpoint",
    "LearningRateLogger",
    "ConsoleLogger",
    "HistoryRecorder",
    "EarlyStopOnNan",
    "TensorBoardCallback",
    "WandbCallback",
    "ExperimentLogger",
    # Experiments / ablations
    "Experiment",
    "ExperimentReport",
    "run_experiment",
    "AblationRunner",
    "AblationReport",
    "ABLATION_VARIANTS",
    "DEFAULT_VARIANTS",
    "build_variant_config",
    # Validation strategies
    "FoldGenerator",
    "build_fold_generator",
    "cross_validation_splits",
    "HoldOutFoldGenerator",
    "KFoldFoldGenerator",
    "StratifiedKFoldFoldGenerator",
    "SpatialFoldGenerator",
    "TemporalFoldGenerator",
    # Visualization
    "Visualizer",
    # Interfaces
    "SchedulerHandle",
    "Weighter",
    # Utilities
    "set_seed",
    "configure_determinism",
    "resolve_device",
    "to_device",
    "setup_distributed",
    "cleanup_distributed",
    "is_distributed",
    "is_primary",
    "get_git_hash",
    "get_git_branch",
    "get_environment_info",
    "apply_gradient_checkpointing",
    "compute_grad_norm",
    # Exceptions
    "TrainingError",
    "TrainingConfigurationError",
    "TrainingRunError",
    "OptimizerBuildError",
    "SchedulerBuildError",
    "LossBuildError",
    "MetricError",
    "CheckpointError",
    "ValidationError",
    "BenchmarkError",
    "VisualizationError",
    "AblationError",
]
