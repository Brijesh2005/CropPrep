"""Training configuration for the CropFusion training engine.

Everything is configurable through YAML (or ``TRN_*`` env vars), mirroring
the resolution order of the other CropFusion packages:

    env (``TRN_<SECTION>__<KEY>``) > YAML (``TRN_CONFIG_FILE``) > defaults

Every field is validated by pydantic. The root :class:`TrainingConfig`
contains one section per subsystem (optimizer, scheduler, loss, checkpoint,
logging, validation, ablation, benchmark, visualization) so a full training
run — including an ablation sweep — can be expressed without writing code.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field

from shared.config import apply_case_insensitive, deep_merge, parse_env
from shared.utils import yaml_safe

from .exceptions import TrainingConfigurationError

ENV_PREFIX = "TRN_"

#: Repository root (parent of the ``training`` package). The shipped config
#: writes artifact paths relative to the repository root so the same YAML
#: works on a research machine and inside a Kaggle notebook.
REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_repo_relative(value: str | Path | None) -> Path | None:
    """Resolve a repository-relative path against :data:`REPO_ROOT`.

    Absolute / root-anchored paths (and ``None``) pass through unchanged;
    relative paths are anchored at the repository root instead of the
    process CWD.
    """
    if value is None:
        return None
    candidate = Path(value)
    if candidate.anchor:
        return candidate
    return (REPO_ROOT / candidate).resolve()


def _str(value: Path | None) -> str | None:
    """Render a resolved path back to ``str`` (None passes through)."""
    return str(value) if value is not None else None


# --------------------------------------------------------------------------- #
# Per-subsystem config sections
# --------------------------------------------------------------------------- #


class GeneralConfig(BaseModel):
    """Seed / device / precision / gradient handling."""

    model_config = ConfigDict(extra="forbid")

    #: ``auto`` (best available) | ``cpu`` | ``cuda``. ``cuda`` requested on a
    #: machine without CUDA falls back to CPU with a warning.
    device: str = "auto"
    #: Root directory for experiment artifacts (runs / ablation / benchmark).
    output_dir: Path = Field(default=Path("artifacts/training"))
    #: Seed used for torch / numpy / python randomness.
    seed: int = 42
    #: Enforce deterministic algorithms (may reduce performance).
    deterministic: bool = False
    #: Automatic mixed precision (fp16 on CUDA, no-op elsewhere).
    amp: bool = False
    #: AMP compute dtype: float16 | bfloat16.
    amp_dtype: str = Field(default="float16", pattern="^(float16|bfloat16)$")
    #: Max gradient norm for clipping (``None`` = disabled).
    gradient_clip: float | None = Field(default=None, ge=0.0)
    #: ``norm`` (clamp total norm) | ``value`` (clamp each param).
    gradient_clip_type: str = Field(default="norm", pattern="^(norm|value)$")
    #: Accumulate gradients over N micro-batches before one optimizer step.
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    #: Trade memory for compute by recomputing activations (image encoders).
    gradient_checkpointing: bool = False
    #: Detect NaN / Inf losses and gradients.
    nan_detection: bool = True
    #: ``warn`` | ``skip`` (skip the step) | ``stop`` (halt training).
    #: ``stop`` is the default so NaN / Inf instability is never silently
    #: skipped — it surfaces as a hard failure with diagnostics (R5.2).
    nan_policy: str = Field(default="stop", pattern="^(warn|skip|stop)$")
    #: Log training metrics every N optimizer steps.
    log_every: int = Field(default=1, ge=1)
    #: Run validation every N epochs.
    validation_frequency: int = Field(default=1, ge=1)
    #: Compile the model with ``torch.compile`` when it is available.
    compile: bool = False
    #: ``default`` | ``reduce-overhead`` | ``max-autotune`` |
    #: ``max-autotune-no-cudagraphs``.
    compile_mode: str = Field(
        default="default",
        pattern="^(default|reduce-overhead|max-autotune|max-autotune-no-cudagraphs)$",
    )
    #: Explicit compile backend (``inductor`` default; ``eager`` useful for
    #: testing). ``None`` lets ``torch.compile`` pick its default.
    compile_backend: str | None = None
    #: Generate the end-of-run reports (training / validation / metrics /
    #: checkpoint / learning-curve).
    reports: bool = True
    #: Report output directory (defaults to ``<output_dir>/reports``).
    reports_dir: str | None = None


class DataConfig(BaseModel):
    """DataLoader settings for the training / validation / test loaders."""

    model_config = ConfigDict(extra="forbid")

    batch_size: int = Field(default=32, ge=1)
    workers: int = Field(default=0, ge=0)
    pin_memory: bool = False
    prefetch_factor: int | None = Field(default=None, ge=1)
    persistent_workers: bool = False
    drop_last: bool = False
    train_shuffle: bool = True


class OptimizerConfig(BaseModel):
    """Optimizer selection and hyper-parameters."""

    model_config = ConfigDict(extra="forbid")

    #: adamw | sgd | radam | lion (validated at build time by
    #: :func:`ai.training.optimizers.build_optimizer`).
    name: str = "adamw"
    lr: float = Field(default=1e-4, gt=0.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    #: adamw / radam / lion betas.
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = Field(default=1e-8, gt=0.0)
    #: sgd momentum (ignored by other optimizers).
    momentum: float = Field(default=0.0, ge=0.0)
    #: sgd nesterov (requires momentum > 0).
    nesterov: bool = False
    #: Lion betas (only used when ``name == lion``).
    lion_beta1: float = Field(default=0.9, ge=0.0, le=1.0)
    lion_beta2: float = Field(default=0.99, ge=0.0, le=1.0)
    #: Discriminative LR: image-backbone parameters train at
    #: ``lr * backbone_lr_multiplier`` while heads / fusion train at ``lr``
    #: (R5.4). ``None`` (default) = a single uniform parameter group. Only
    #: applies to models exposing ``ndvi_encoder`` / ``evi_encoder``
    #: backbones (CropFusionModel).
    backbone_lr_multiplier: float | None = Field(default=None, gt=0.0, le=1.0)


class SchedulerConfig(BaseModel):
    """LR scheduler selection and hyper-parameters."""

    model_config = ConfigDict(extra="forbid")

    #: none | cosine | onecycle | reduce_on_plateau | polynomial |
    #: warmup_cosine | warmup_polynomial.
    name: str = Field(
        default="cosine",
        pattern=(
            "^(none|cosine|onecycle|reduce_on_plateau|polynomial|"
            "warmup_cosine|warmup_polynomial)$"
        ),
    )
    #: Step the scheduler once per epoch or once per optimizer step.
    step: str = Field(default="epoch", pattern="^(epoch|step)$")
    #: Warmup length (steps or epochs matching ``step``).
    warmup_steps: int = Field(default=0, ge=0)
    warmup_epochs: int = Field(default=0, ge=0)
    #: Warmup as a fraction of the schedule length (overrides the above).
    warmup_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    #: Cosine / polynomial: number of cycles in epochs (``None`` = epochs).
    t_max: int | None = Field(default=None, ge=1)
    #: Cosine minimum LR.
    eta_min: float = Field(default=0.0, ge=0.0)
    #: OneCycle progress at peak LR.
    pct_start: float = Field(default=0.3, ge=0.0, le=1.0)
    #: OneCycle initial LR = max_lr / div_factor.
    div_factor: float = Field(default=25.0, gt=0.0)
    #: OneCycle final LR = max_lr / (div_factor * final_div_factor).
    final_div_factor: float = Field(default=1e4, gt=0.0)
    #: ReduceLROnPlateau factor / patience / threshold / cooldown.
    factor: float = Field(default=0.1, gt=0.0, le=1.0)
    patience: int = Field(default=10, ge=0)
    threshold: float = Field(default=1e-4, ge=0.0)
    cooldown: int = Field(default=0, ge=0)
    min_lr: float = Field(default=0.0, ge=0.0)
    mode: str = Field(default="min", pattern="^(min|max)$")
    #: Polynomial decay exponent + final LR.
    power: float = Field(default=1.0, gt=0.0)
    end_lr: float = Field(default=0.0, ge=0.0)


class LossConfig(BaseModel):
    """Multi-task loss settings."""

    model_config = ConfigDict(extra="forbid")

    #: crop: cross_entropy | label_smoothing | focal.
    crop_loss: str = Field(
        default="label_smoothing",
        pattern="^(cross_entropy|label_smoothing|focal)$",
    )
    #: yield: mse | huber | mae.
    yield_loss: str = Field(default="huber", pattern="^(mse|huber|mae)$")
    #: Fixed task weights (used when ``weighting_mode == fixed``).
    crop_weight: float = Field(default=0.7, ge=0.0)
    yield_weight: float = Field(default=0.3, ge=0.0)
    #: fixed | uncertainty (Kendall) | gradnorm.
    weighting_mode: str = Field(
        default="fixed", pattern="^(fixed|uncertainty|gradnorm)$"
    )
    label_smoothing: float = Field(default=0.1, ge=0.0, lt=1.0)
    focal_gamma: float = Field(default=2.0, ge=0.0)
    reduction: str = Field(default="mean", pattern="^(mean|sum)$")
    #: GradNorm asymmetry parameter ``alpha`` (Chen et al., 2018).
    gradnorm_alpha: float = Field(default=1.5, ge=0.0)
    #: Floor for learnable log-variances (numerical stability).
    log_variance_eps: float = Field(default=0.01, ge=1e-6)
    #: Class-imbalance weighting for the crop task: ``none`` | ``balanced`` |
    #: ``sqrt_inv`` | ``effective_num``. Weights are derived from the training
    #: set's class frequencies (no oversampling; focal loss is future work).
    class_weight_mode: str = Field(
        default="none",
        pattern="^(none|balanced|sqrt_inv|effective_num)$",
    )
    #: Floor for class-frequency weights (numerical stability).
    class_weight_eps: float = Field(default=1e-6, ge=0.0)
    #: Effective-number beta (only used when ``class_weight_mode ==
    #: effective_num``).
    class_weight_beta: float = Field(default=0.999, gt=0.0, lt=1.0)


class TrainConfig(BaseModel):
    """Epoch count and early-stopping behaviour."""

    model_config = ConfigDict(extra="forbid")

    epochs: int = Field(default=100, ge=1)
    #: Metric watched by early stopping (``val_loss`` by default).
    early_stopping_metric: str = "val_loss"
    #: ``min`` (lower is better) | ``max`` (higher is better).
    early_stopping_mode: str = Field(default="min", pattern="^(min|max)$")
    #: Epochs of no improvement before stopping (``None`` = disabled).
    early_stopping_patience: int | None = Field(default=10, ge=0)
    #: Minimum improvement required to reset the counter.
    early_stopping_min_delta: float = Field(default=0.0, ge=0.0)
    #: Restore the best weights when early stopping triggers.
    restore_best_on_stop: bool = True


class CheckpointConfig(BaseModel):
    """Checkpoint policy."""

    model_config = ConfigDict(extra="forbid")

    directory: str = "artifacts/training/checkpoints"
    #: Keep this many most-recent checkpoints (``None`` = keep all).
    keep_last: int | None = Field(default=3, ge=1)
    save_best: bool = True
    save_latest: bool = True
    #: Save a periodic checkpoint every N epochs (``None`` = disabled).
    save_periodic: int | None = Field(default=None, ge=1)
    #: Automatically resume from the latest checkpoint in ``directory``.
    resume: bool = False
    #: Explicit checkpoint file to resume from (overrides ``resume`` discovery).
    resume_path: str | None = None


class MetricsConfig(BaseModel):
    """Evaluation metric configuration."""

    model_config = ConfigDict(extra="forbid")

    #: Top-K accuracy for classification.
    top_k: int = Field(default=5, ge=1)
    #: macro | micro | weighted averaging for precision/recall/f1.
    average: str = Field(default="macro", pattern="^(macro|micro|weighted)$")
    #: Compute ROC-AUC (one-vs-rest). Expensive for many classes.
    roc_auc: bool = False
    #: Report per-class precision/recall/f1 in the evaluation output.
    per_class: bool = False


class LoggingConfig(BaseModel):
    """Experiment logging / tracking settings."""

    model_config = ConfigDict(extra="forbid")

    level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR)$")
    console: bool = True
    csv: bool = True
    #: Write structured per-epoch metrics as JSON.
    json_logs: bool = True
    #: TensorBoard writer (only used when the package is installed).
    tensorboard: bool = False
    tensorboard_dir: str = "artifacts/training/tensorboard"
    #: Weights & Biases (only used when the package is installed).
    wandb: bool = False
    wandb_project: str = "cropfusion"
    wandb_entity: str | None = None
    #: Snapshot the resolved config (training + model + preprocessing) to disk.
    config_snapshot: bool = True
    #: Record the git commit hash when the repo is available.
    git_hash: bool = True


class ValidationConfig(BaseModel):
    """Model-validation strategy (hold-out / cross-validation)."""

    model_config = ConfigDict(extra="forbid")

    #: holdout | kfold | stratified_kfold | spatial | temporal.
    strategy: str = Field(
        default="holdout",
        pattern="^(holdout|kfold|stratified_kfold|spatial|temporal)$",
    )
    #: Number of folds (kfold / stratified_kfold / spatial / temporal).
    k_folds: int = Field(default=5, ge=2)
    #: Shuffle observations before fold assignment.
    shuffle: bool = True
    seed: int = 42
    #: Attribute used for spatial / group folds (e.g. "village").
    group_column: str = "village"
    #: Attribute used for temporal folds (e.g. "year").
    temporal_column: str = "year"
    #: Enable fp16 autocast during validation passes. FP32 validation is the
    #: default: the eval fast path encodes ``B * T`` frames in one backbone
    #: forward (e.g. 16 x 8 = 128 frames), and fp16 GEMM/accumulation can go
    #: non-finite there (R5.3 TR-VAL-001 at ``ndvi_encoder.backbone.blocks.4.2``
    #: on P100) — training AMP is controlled separately by ``general.amp``.
    #: Keep this ``False`` unless a numerics probe proves fp16 validation is
    #: finite for the deployed batch size / GPU pair.
    amp: bool = False
    #: AMP compute dtype for validation when ``amp`` is enabled.
    amp_dtype: str = Field(default="float16", pattern="^(float16|bfloat16)$")


class AblationConfig(BaseModel):
    """Ablation sweep settings."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    #: full | only_tabular | only_ndvi | only_evi | only_image |
    #: no_cross_attention | no_adaptive_gate.
    variants: list[str] = Field(
        default_factory=lambda: [
            "full",
            "only_tabular",
            "only_ndvi",
            "only_evi",
            "only_image",
            "no_cross_attention",
            "no_adaptive_gate",
        ]
    )
    #: Metric used to rank variants in the comparison report.
    compare_metric: str = "multi_task_score"
    compare_mode: str = Field(default="max", pattern="^(min|max)$")


class BenchmarkConfig(BaseModel):
    """Inference / training benchmark settings."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    #: Repeat passes for the latency distribution.
    iterations: int = Field(default=100, ge=1)
    warmup_iterations: int = Field(default=10, ge=0)
    batch_size: int = Field(default=32, ge=1)
    measure_training_speed: bool = True
    measure_inference_speed: bool = True
    #: Only time a forward pass (skip building the optimizer etc.).
    inference_only: bool = False


class VisualizationConfig(BaseModel):
    """Automatic visualization settings."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    directory: str = "artifacts/training/visualizations"
    dashboard: bool = True
    loss_curves: bool = True
    accuracy_curves: bool = True
    lr_curves: bool = True
    regression_scatter: bool = True
    confusion_matrix: bool = True
    precision_recall: bool = True
    feature_distribution: bool = True


# --------------------------------------------------------------------------- #
# Root config
# --------------------------------------------------------------------------- #


class FineTuningStage(BaseModel):
    """One progressive-unfreeze step of the image backbone.

    At ``epoch`` the parameters whose full dotted names match one of
    ``prefixes`` are unfrozen (``requires_grad_`` True). Prefixes match a
    module segment, e.g. ``blocks.6`` unfreezes ``ndvi_encoder.backbone.
    blocks.6.*`` and ``evi_encoder.backbone.blocks.6.*``. A stage listing an
    empty ``prefixes`` list is documented-only (freeze-everything stage).
    """

    model_config = ConfigDict(extra="forbid")

    epoch: int = Field(ge=0)
    prefixes: list[str] = Field(default_factory=list)


class FineTuningConfig(BaseModel):
    """Staged backbone fine-tuning (R5.4, EfficientNet blocks).

    Complements (and is independent of) the ``curriculum``:
    :class:`Curriculum` freezes whole *modalities* (tabular / image /
    temporal / fusion) by training stage, while fine-tuning here progressively
    unfreezes the pretrained image backbones by epoch. When both are enabled
    the fine-tuning stage callback only starts unfreezing once the curriculum
    has reached a stage where the image branch is trainable; to avoid clashes
    we recommend enabling either ``curriculum`` *or* ``fine_tuning`` per run.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    schedule: list[FineTuningStage] = Field(default_factory=list)


class CurriculumConfig(BaseModel):
    """Five-stage curriculum training settings.

    Stage schedule::

        1. tabular  — train the Tabular Encoder only.
        2. image    — train the image encoders (NDVI / EVI + image fusion).
        3. temporal — train the Temporal Encoder.
        4. fusion   — train the Fusion Engine (cross attention / gated fusion /
                      shared encoder).
        5. finetune — fine-tune the entire network.

    Transitions are automatic: each active stage runs for its share of the
    epoch budget (``epochs_per_stage`` overrides individual stages; missing
    stages are filled from the remaining budget, otherwise the total is split
    evenly across the active stages). ``start_stage`` skips earlier stages,
    which doubles as resume-from-any-stage.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    #: First stage to run (1..5); earlier stages are skipped (resume semantics).
    start_stage: int = Field(default=1, ge=1, le=5)
    #: Optional explicit per-stage epoch counts keyed by stage name
    #: (``tabular`` | ``image`` | ``temporal`` | ``fusion`` | ``finetune``).
    #: Unspecified stages receive the remaining epoch budget.
    epochs_per_stage: dict[str, int] | None = None
    #: Record the active stage name into each epoch's logs / history.
    log_transitions: bool = True


class DataContractConfig(BaseModel):
    """Training-data contract gate (R5.2.1 Task D).

    A training run is refused up front when the corpus violates the contract
    (mixed yield units, or a crop classifier enabled without crop labels)
    instead of silently training on an invalid target. Disabling the gate is
    allowed only for explicit diagnostics runs.
    """

    model_config = ConfigDict(extra="forbid")

    #: Validate the training corpus against the data contract before training.
    enabled: bool = True
    #: ``True`` = raise on hard violations; ``False`` = log a warning only.
    strict: bool = True


class TrainingConfig(BaseModel):
    """Root training configuration."""

    model_config = ConfigDict(extra="forbid")

    name: str = "cropfusion_training"
    general: GeneralConfig = Field(default_factory=GeneralConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    data_contract: DataContractConfig = Field(default_factory=DataContractConfig)
    curriculum: CurriculumConfig = Field(default_factory=CurriculumConfig)
    fine_tuning: FineTuningConfig = Field(default_factory=FineTuningConfig)
    ablation: AblationConfig = Field(default_factory=AblationConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)

    def to_yaml(self) -> str:
        return yaml.safe_dump(yaml_safe(self.model_dump()), sort_keys=False)

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.write_text(self.to_yaml(), encoding="utf-8")
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrainingConfig":
        return cls.model_validate(dict(data))


def load_training_config(
    config_path: str | Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> TrainingConfig:
    """Load and validate training settings (env > YAML > defaults).

    Args:
        config_path: Path to a YAML training config.
        env: Optional environment mapping (defaults to ``os.environ``).

    Raises:
        TrainingConfigurationError: When the file is missing, malformed or
            invalid.
    """
    env_map = dict(os.environ if env is None else env)
    if config_path is None:
        env_config = env_map.get("TRN_CONFIG_FILE")
        config_path = env_config or None

    data: dict[str, Any] = {}
    if config_path is not None:
        config_file = Path(config_path)
        if not config_file.exists():
            raise TrainingConfigurationError(
                f"Training config file not found: {config_file}",
                detail=str(config_file),
            )
        try:
            raw = yaml.safe_load(config_file.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise TrainingConfigurationError(
                f"Malformed training YAML: {exc}", detail=str(config_file)
            ) from exc
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise TrainingConfigurationError("Training config root must be a mapping")
        data = raw

    parsed_env = parse_env(env_map, prefix=ENV_PREFIX)
    # ``TRN_CONFIG_FILE`` selects the YAML; it is not a config field.
    parsed_env.pop("config_file", None)
    merged = deep_merge(data, parsed_env)
    merged = apply_case_insensitive(merged, TrainingConfig)
    try:
        config = TrainingConfig.model_validate(merged)
    except Exception as exc:  # pydantic.ValidationError
        raise TrainingConfigurationError(f"Invalid training configuration: {exc}") from exc

    # Artifact paths in the shipped config are repository-relative; anchor them
    # at the repository root so resolution does not depend on the CWD.
    config.general.output_dir = _resolve_repo_relative(config.general.output_dir)
    config.general.reports_dir = _str(_resolve_repo_relative(config.general.reports_dir))
    config.checkpoint.directory = _str(_resolve_repo_relative(config.checkpoint.directory))
    config.logging.tensorboard_dir = _str(
        _resolve_repo_relative(config.logging.tensorboard_dir)
    )
    config.visualization.directory = _str(
        _resolve_repo_relative(config.visualization.directory)
    )
    return config


def save_training_template(path: str | Path) -> Path:
    """Write an annotated YAML template of the default training config."""
    template = TrainingConfig().model_dump()
    out = Path(path)
    out.write_text(
        yaml.safe_dump(yaml_safe(template), sort_keys=False), encoding="utf-8"
    )
    return out
