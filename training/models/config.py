"""Model configuration for the CropFusion neural architecture.

Everything is configurable through YAML (or ``MODEL_CONFIG_FILE`` env), with
defaults matching the Phase 5 specification. Settings resolve env
(``MODEL_``) > YAML > defaults, and every field is validated by pydantic —
mirroring the resolution order of the preprocessing config.

The tabular schema (``numeric_dim`` + ``categorical_cardinalities``) and the
crop head's ``num_classes`` are derived from a fitted Phase 4
:class:`~ai.preprocessing.master_pipeline.Preprocessor` via
:meth:`ModelConfig.from_preprocessor`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from shared.config import apply_case_insensitive, deep_merge, parse_env

from .exceptions import ModelConfigurationError

ENV_PREFIX = "MODEL_"

#: Activation choices accepted across the configuration.
_ACTIVATIONS = ("relu", "gelu", "silu", "tanh", "leaky_relu")


class TabularModelConfig(BaseModel):
    """TabTransformer settings."""

    model_config = ConfigDict(extra="forbid")

    #: Number of leading continuous features (see module docstring layout).
    numeric_dim: int = Field(default=0, ge=0)
    #: Cardinality (number of seen categories) of each categorical feature,
    #: in the same order as the Phase 4 ordinal encoder columns.
    categorical_cardinalities: list[int] = Field(default_factory=list)
    #: Transformer embedding dimension (all tokens share this width).
    embedding_dim: int = Field(default=64, ge=8)
    #: Number of transformer encoder blocks.
    depth: int = Field(default=4, ge=1)
    #: Attention heads per block.
    num_heads: int = Field(default=4, ge=1)
    #: Feed-forward hidden width.
    ff_dim: int = Field(default=256, ge=8)
    #: Dropout applied throughout the stack.
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    activation: str = Field(default="gelu", pattern="|".join(_ACTIVATIONS))
    #: Prepend a learnable CLS token and pool it for the tabular embedding.
    use_cls: bool = True
    #: none | sinusoidal | learned.
    position_encoding: str = Field(default="none", pattern="^(none|sinusoidal|learned)$")
    #: Upper bound on the number of feature tokens (embedding/positional).
    max_len: int = Field(default=64, ge=1)

    @field_validator("num_heads")
    @classmethod
    def _heads_dividing(cls, value: int, info: Any) -> int:
        if "embedding_dim" in info.data and info.data["embedding_dim"] % value != 0:
            raise ValueError("num_heads must divide embedding_dim")
        return value


class ImageEncoderConfig(BaseModel):
    """Shared settings for the NDVI and EVI timm backbones."""

    model_config = ConfigDict(extra="forbid")

    #: timm backbone name (default EfficientNetV2-S); ``None`` disables the
    #: image branch.
    backbone: str | None = "efficientnetv2_s"
    #: Load ImageNet pretrained weights (requires network on first run).
    pretrained: bool = False
    #: Freeze all backbone weights after construction.
    freeze_backbone: bool = False
    #: Square edge the patches are resized to before the backbone (``None`` =
    #: use the backbone's native resolution).
    input_size: int | None = Field(default=None, ge=8)
    #: How single-channel patches become 3-channel: repeat | conv.
    channel_expansion: str = Field(default="repeat", pattern="^(repeat|conv)$")
    #: Stochastic depth rate for the backbone (0 disables).
    drop_path_rate: float = Field(default=0.0, ge=0.0)
    #: Per-modality backbone overrides (fall back to ``backbone``).
    ndvi_backbone: str | None = None
    evi_backbone: str | None = None
    #: Enable the NDVI encoder (ablation support). At least one of
    #: ``enable_ndvi`` / ``enable_evi`` must be on when ``backbone`` is set.
    enable_ndvi: bool = True
    #: Enable the EVI encoder (ablation support).
    enable_evi: bool = True

    @model_validator(mode="after")
    def _at_least_one_stream(self) -> "ImageEncoderConfig":
        if self.backbone is not None and not (self.enable_ndvi or self.enable_evi):
            raise ValueError(
                "image_encoder requires at least one stream: set enable_ndvi "
                "or enable_evi (or backbone=None to disable the image branch)"
            )
        return self


class ImageFusionConfig(BaseModel):
    """Per-timestep NDVI/EVI feature fusion settings."""

    model_config = ConfigDict(extra="forbid")

    #: concat | weighted_sum | learnable | attention.
    method: str = Field(
        default="learnable",
        pattern="^(concat|weighted_sum|learnable|attention)$",
    )
    #: Fusion working/output width (``None`` = ``min(encoder width, 512)``).
    hidden_dim: int | None = Field(default=None, ge=8)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)


class TemporalModelConfig(BaseModel):
    """Temporal transformer settings (variable-length sequence encoding)."""

    model_config = ConfigDict(extra="forbid")

    d_model: int = Field(default=256, ge=8)
    depth: int = Field(default=2, ge=1)
    num_heads: int = Field(default=4, ge=1)
    ff_dim: int = Field(default=1024, ge=8)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    activation: str = Field(default="gelu", pattern="|".join(_ACTIVATIONS))
    use_cls: bool = True
    #: none | sinusoidal | learned.
    position_encoding: str = Field(default="learned", pattern="^(none|sinusoidal|learned)$")
    #: Upper bound on the number of timesteps (must cover Phase 4's
    #: ``temporal.max_observations``).
    max_len: int = Field(default=16, ge=1)
    #: Output image-embedding width (post pooling).
    embedding_dim: int = Field(default=256, ge=8)

    @field_validator("num_heads")
    @classmethod
    def _heads_dividing(cls, value: int, info: Any) -> int:
        if "d_model" in info.data and info.data["d_model"] % value != 0:
            raise ValueError("num_heads must divide d_model")
        return value


class CrossAttentionConfig(BaseModel):
    """Cross-modal attention settings (image queries tabular)."""

    model_config = ConfigDict(extra="forbid")

    #: Whether the cross-attention block is active (ablation support).
    enabled: bool = True
    num_heads: int = Field(default=4, ge=1)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    #: Width of the cross-attention output.
    out_dim: int = Field(default=256, ge=8)


class GatedFusionConfig(BaseModel):
    """Adaptive gated-fusion settings (image / tabular / fusion gates)."""

    model_config = ConfigDict(extra="forbid")

    #: Whether the adaptive gated-fusion block is active (ablation support).
    #: When disabled the image + tabular (+ cross) representations are
    #: concatenated into the shared encoder instead.
    enabled: bool = True
    out_dim: int = Field(default=256, ge=8)
    hidden_dim: int = Field(default=256, ge=8)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)


class SharedEncoderConfig(BaseModel):
    """Shared multimodal latent encoder settings."""

    model_config = ConfigDict(extra="forbid")

    d_model: int = Field(default=256, ge=8)
    depth: int = Field(default=2, ge=1)
    num_heads: int = Field(default=4, ge=1)
    ff_dim: int = Field(default=1024, ge=8)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    activation: str = Field(default="gelu", pattern="|".join(_ACTIVATIONS))
    #: Width of the shared multimodal representation (512 / 768 / 1024 ...).
    out_dim: int = Field(default=512, ge=8)

    @field_validator("num_heads")
    @classmethod
    def _heads_dividing(cls, value: int, info: Any) -> int:
        if "d_model" in info.data and info.data["d_model"] % value != 0:
            raise ValueError("num_heads must divide d_model")
        return value


class FusionConfig(BaseModel):
    """Cross-modal fusion-engine behaviour (Phase 6 ``CrossModalFusionEngine``).

    The engine owns the cross-attention block, the adaptive gated fusion and
    the shared multimodal encoder, so the whole cross-modal pathway is one
    swappable unit (see :class:`~ai.models.fusion_engine.CrossModalFusionEngine`).
    """

    model_config = ConfigDict(extra="forbid")

    #: Add the projected modality streams back to the gated fusion output as a
    #: residual (the shared encoder then attends over a gated-plus-original
    #: token). Disable for the pure gated ablation.
    residual_fusion: bool = True
    #: Feed the temporal image embedding into the gated fusion as a fourth
    #: stream (adds a ``temporal_gate``). Requires the image branch (the
    #: temporal transformer output). Ablation-off by default to preserve the
    #: Phase 5 image / tabular / fusion gate contract.
    use_temporal_stream: bool = False


class RuntimeConfig(BaseModel):
    """Execution-time settings applied when the model is created / deployed.

    These are handled by :mod:`ai.models.runtime` and never touch the
    architecture itself: precision (AMP), device placement, ``torch.compile``,
    gradient checkpointing and single-node / distributed data parallelism.
    """

    model_config = ConfigDict(extra="forbid")

    #: Parameter / activation dtype: float32 | float16 | bfloat16.
    precision: str = Field(
        default="float32", pattern="^(float32|float16|bfloat16)$"
    )
    #: Explicit device (``cpu`` / ``cuda`` / ``cuda:0`` / ``mps``); ``None`` =
    #: auto (CUDA if available else CPU).
    device: str | None = None
    #: Compile the model with ``torch.compile`` at creation time.
    compile: bool = False
    #: ``torch.compile`` mode (ignored unless ``compile`` is set).
    compile_mode: str = Field(
        default="default",
        pattern="^(default|reduce-overhead|max-autotune|max-autotune-no-cudagraphs)$",
    )
    #: Trade activation memory for recomputation on the transformer stacks.
    gradient_checkpointing: bool = False
    #: Wrap the model in ``torch.nn.DataParallel`` (single node, multi GPU).
    data_parallel: bool = False
    #: Wrap the model in ``torch.distributed`` DistributedDataParallel.
    distributed: bool = False
    #: Local rank for distributed wrapping (defaults to
    #: ``torch.distributed.get_rank()``).
    local_rank: int | None = Field(default=None, ge=0)


class CropHeadConfig(BaseModel):
    """Crop recommendation head (multi-class softmax classification)."""

    model_config = ConfigDict(extra="forbid")

    #: Number of crop classes (derived from the Phase 4 label encoder).
    num_classes: int = Field(default=0, ge=0)
    hidden_dim: int | None = Field(default=None, ge=8)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    activation: str = Field(default="relu", pattern="|".join(_ACTIVATIONS))


class YieldHeadConfig(BaseModel):
    """Yield prediction head (single-value regression)."""

    model_config = ConfigDict(extra="forbid")

    hidden_dim: int | None = Field(default=None, ge=8)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    activation: str = Field(default="relu", pattern="|".join(_ACTIVATIONS))
    #: Lower clamp applied to the predicted yield (``None`` = unconstrained).
    output_clamp_min: float | None = None


class HeadsConfig(BaseModel):
    """Multi-task head registry.

    ``crop`` and ``yield_prediction`` are the two built-in heads; future heads
    (health / disease / water requirement) register through the model's
    ``add_head`` API.
    """

    model_config = ConfigDict(extra="forbid")

    crop: CropHeadConfig | None = Field(default_factory=CropHeadConfig)
    yield_prediction: YieldHeadConfig | None = Field(default_factory=YieldHeadConfig)


class LossConfig(BaseModel):
    """Loss interfaces (never used for training — Phase 6)."""

    model_config = ConfigDict(extra="forbid")

    #: cross_entropy | label_smoothing | focal.
    crop_loss: str = Field(
        default="label_smoothing",
        pattern="^(cross_entropy|label_smoothing|focal)$",
    )
    #: mse | huber.
    yield_loss: str = Field(default="huber", pattern="^(mse|huber)$")
    #: Fixed task weights (used when ``weighting_mode=fixed``).
    crop_weight: float = Field(default=0.7, ge=0.0)
    yield_weight: float = Field(default=0.3, ge=0.0)
    #: fixed | learnable (Kendall-style Gaussian uncertainty weighting).
    weighting_mode: str = Field(default="fixed", pattern="^(fixed|learnable)$")
    label_smoothing: float = Field(default=0.1, ge=0.0, lt=1.0)
    focal_gamma: float = Field(default=2.0, ge=0.0)
    reduction: str = Field(default="mean", pattern="^(mean|sum)$")
    #: Floor for learnable log-variances (numerical stability).
    log_variance_eps: float = Field(default=0.01, ge=1e-6)


class CheckpointConfig(BaseModel):
    """Checkpoint manager defaults."""

    model_config = ConfigDict(extra="forbid")

    directory: str = "artifacts/models"
    #: Number of most-recent checkpoints to keep (``None`` = keep all).
    keep_last: int | None = Field(default=3, ge=1)


class ExportConfig(BaseModel):
    """Export defaults (TorchScript / ONNX / future TensorRT)."""

    model_config = ConfigDict(extra="forbid")

    #: ONNX opset version used by :class:`~ai.models.exporter.ModelExporter`.
    onnx_opset: int = Field(default=17, ge=9, le=22)
    #: torchscript | trace — scripting mode for TorchScript export.
    torchscript_mode: str = Field(default="trace", pattern="^(trace|script)$")


class ModelConfig(BaseModel):
    """Root model configuration."""

    model_config = ConfigDict(extra="forbid")

    name: str = "cropfusion_v1"
    version: str = "1.0.0"
    #: Schema version of the architecture the config describes. Bumped on
    #: breaking architectural changes so checkpoints can be re-validated.
    architecture_version: str = "1.0.0"

    tabular: TabularModelConfig = Field(default_factory=TabularModelConfig)
    image_encoder: ImageEncoderConfig = Field(default_factory=ImageEncoderConfig)
    image_fusion: ImageFusionConfig = Field(default_factory=ImageFusionConfig)
    temporal: TemporalModelConfig = Field(default_factory=TemporalModelConfig)
    cross_attention: CrossAttentionConfig = Field(default_factory=CrossAttentionConfig)
    gated_fusion: GatedFusionConfig = Field(default_factory=GatedFusionConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    shared_encoder: SharedEncoderConfig = Field(default_factory=SharedEncoderConfig)
    heads: HeadsConfig = Field(default_factory=HeadsConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    #: Validate inputs (shape / dtype) inside every forward pass.
    validate_inputs: bool = True

    # -- Derived schema helpers -------------------------------------------- #

    @property
    def uses_tabular(self) -> bool:
        """Whether the tabular branch is enabled."""
        return self.tabular.numeric_dim > 0 or bool(self.tabular.categorical_cardinalities)

    @property
    def uses_image(self) -> bool:
        """Whether the image branch is enabled."""
        return self.image_encoder.backbone is not None

    @property
    def tabular_feature_dim(self) -> int:
        """Total width of the Phase 4 ``[F]`` tabular tensor."""
        return self.tabular.numeric_dim + len(self.tabular.categorical_cardinalities)

    @property
    def crop_enabled(self) -> bool:
        return self.heads.crop is not None and self.heads.crop.num_classes > 0

    @property
    def yield_enabled(self) -> bool:
        return self.heads.yield_prediction is not None

    @model_validator(mode="after")
    def _at_least_one_modality(self) -> "ModelConfig":
        if not self.uses_tabular and not self.uses_image:
            raise ValueError(
                "ModelConfig requires at least one modality: enable tabular "
                "(numeric_dim/categorical_cardinalities) or image_encoder "
                "(backbone)."
            )
        if not self.crop_enabled and not self.yield_enabled:
            raise ValueError(
                "At least one task head must be enabled "
                "(heads.crop.num_classes > 0 and/or heads.yield_prediction)."
            )
        if self.fusion.use_temporal_stream and not self.uses_image:
            raise ValueError(
                "fusion.use_temporal_stream requires the image branch "
                "(image_encoder.backbone must be set)."
            )
        return self

    # -- Construction helpers ----------------------------------------------- #

    @staticmethod
    def _derived_schema(preprocessor: Any) -> dict[str, Any]:
        """Extract the schema fields the preprocessor owns (partial dict).

        * ``tabular`` — ordinal schema (or all-continuous for one-hot).
        * ``heads.crop.num_classes`` — from the fitted label encoder.
        * ``image_encoder.input_size`` — the preprocessor patch size.
        * ``temporal.max_len`` — enough for ``max_observations``.

        Returns a *partial* dict (only derived keys) so callers can merge it
        without clobbering user-provided architecture settings.
        """
        tabular_pipeline = preprocessor.tabular
        label_pipeline = preprocessor.label

        # Ordinal encoding preserves categorical indices (0..C-1, -1 = unseen)
        # so the TabTransformer can embed them. One-hot / no encoding is
        # consumed as a fully continuous vector instead.
        from training.preprocessing.transforms import OrdinalEncoder

        encoder = getattr(tabular_pipeline, "encoder", None)
        if isinstance(encoder, OrdinalEncoder):
            numeric_dim = len(tabular_pipeline.numeric_features)
            cardinalities = [
                len(categories) for categories in getattr(encoder, "categories_", [])
            ]
        else:
            numeric_dim = len(tabular_pipeline.feature_names)
            cardinalities = []

        num_classes = int(getattr(label_pipeline, "num_classes", 0) or 0)
        image_size = int(getattr(preprocessor.config.image, "size", 128))
        max_observations = int(
            getattr(preprocessor.config.temporal, "max_observations", 8)
        )

        return {
            "tabular": {
                "numeric_dim": numeric_dim,
                "categorical_cardinalities": cardinalities,
            },
            "image_encoder": {"input_size": image_size},
            "temporal": {"max_len": max(16, max_observations)},
            "heads": {"crop": {"num_classes": num_classes}},
        }

    @classmethod
    def from_preprocessor(
        cls, preprocessor: Any, **overrides: Any
    ) -> "ModelConfig":
        """Derive a model config from a fitted Phase 4 :class:`Preprocessor`.

        Infers the tabular schema, crop class count and default image size
        from the fitted preprocessing pipelines, so the model consumes
        Phase 4 output without manual bookkeeping.

        Args:
            preprocessor: Fitted :class:`~ai.preprocessing.master_pipeline.
                Preprocessor`.
            **overrides: Any ``ModelConfig`` field to override after
                derivation (e.g. ``tabular=...``, ``image_encoder=...``).

        Returns:
            A validated :class:`ModelConfig`.
        """
        derived = cls._derived_schema(preprocessor)
        merged = deep_merge(derived, overrides)
        return cls(**merged)

    # -- Serialization ------------------------------------------------------ #

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(), sort_keys=False)

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.write_text(self.to_yaml(), encoding="utf-8")
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelConfig":
        return cls.model_validate(dict(data))


def load_model_config(
    config_path: str | Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> ModelConfig:
    """Load and validate model settings (env > YAML > defaults).

    Args:
        config_path: Path to a YAML model config.
        env: Optional environment mapping (defaults to ``os.environ``).

    Raises:
        ModelConfigurationError: When the file is missing, malformed or
            invalid.
    """
    env_map = dict(os.environ if env is None else env)
    if config_path is None:
        config_path = env_map.get("MODEL_CONFIG_FILE")

    data: dict[str, Any] = {}
    if config_path is not None:
        config_file = Path(config_path)
        if not config_file.exists():
            raise ModelConfigurationError(
                f"Model config file not found: {config_file}", detail=str(config_file)
            )
        try:
            raw = yaml.safe_load(config_file.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ModelConfigurationError(
                f"Malformed model YAML: {exc}", detail=str(config_file)
            ) from exc
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ModelConfigurationError("Model config root must be a mapping")
        data = raw

    parsed_env = parse_env(env_map, prefix=ENV_PREFIX)
    # ``MODEL_CONFIG_FILE`` selects the YAML, it is not a config field —
    # drop the derived ``config_file`` key so validation doesn't reject it.
    parsed_env.pop("config_file", None)
    merged = deep_merge(data, parsed_env)
    merged = apply_case_insensitive(merged, ModelConfig)
    try:
        return ModelConfig.model_validate(merged)
    except Exception as exc:  # pydantic.ValidationError
        raise ModelConfigurationError(f"Invalid model configuration: {exc}") from exc


def save_model_template(path: str | Path) -> Path:
    """Write an annotated YAML template of the default configuration."""
    template = {
        "name": "cropfusion_v1",
        "version": "1.0.0",
        "architecture_version": "1.0.0",
        "tabular": {
            "numeric_dim": 0,
            "categorical_cardinalities": [],
            "embedding_dim": 64,
            "depth": 4,
            "num_heads": 4,
            "ff_dim": 256,
            "dropout": 0.1,
            "activation": "gelu",
            "use_cls": True,
            "position_encoding": "none",
            "max_len": 64,
        },
        "image_encoder": {
            "backbone": "efficientnetv2_s",
            "pretrained": False,
            "freeze_backbone": False,
            "input_size": None,
            "channel_expansion": "repeat",
            "drop_path_rate": 0.0,
            "ndvi_backbone": None,
            "evi_backbone": None,
            "enable_ndvi": True,
            "enable_evi": True,
        },
        "image_fusion": {"method": "learnable", "hidden_dim": None, "dropout": 0.1},
        "temporal": {
            "d_model": 256,
            "depth": 2,
            "num_heads": 4,
            "ff_dim": 1024,
            "dropout": 0.1,
            "activation": "gelu",
            "use_cls": True,
            "position_encoding": "learned",
            "max_len": 16,
            "embedding_dim": 256,
        },
        "cross_attention": {"enabled": True, "num_heads": 4, "dropout": 0.1, "out_dim": 256},
        "gated_fusion": {"enabled": True, "out_dim": 256, "hidden_dim": 256, "dropout": 0.1},
        "fusion": {"residual_fusion": True, "use_temporal_stream": False},
        "shared_encoder": {
            "d_model": 256,
            "depth": 2,
            "num_heads": 4,
            "ff_dim": 1024,
            "dropout": 0.1,
            "activation": "gelu",
            "out_dim": 512,
        },
        "heads": {
            "crop": {"num_classes": 0, "hidden_dim": None, "dropout": 0.1,
                     "activation": "relu"},
            "yield_prediction": {"hidden_dim": None, "dropout": 0.1,
                                 "activation": "relu", "output_clamp_min": None},
        },
        "loss": {
            "crop_loss": "label_smoothing",
            "yield_loss": "huber",
            "crop_weight": 0.7,
            "yield_weight": 0.3,
            "weighting_mode": "fixed",
            "label_smoothing": 0.1,
            "focal_gamma": 2.0,
            "reduction": "mean",
            "log_variance_eps": 0.01,
        },
        "checkpoint": {"directory": "artifacts/models", "keep_last": 3},
        "export": {"onnx_opset": 17, "torchscript_mode": "trace"},
        "runtime": {
            "precision": "float32",
            "device": None,
            "compile": False,
            "compile_mode": "default",
            "gradient_checkpointing": False,
            "data_parallel": False,
            "distributed": False,
            "local_rank": None,
        },
        "validate_inputs": True,
    }
    out = Path(path)
    out.write_text(yaml.safe_dump(template, sort_keys=False), encoding="utf-8")
    return out
