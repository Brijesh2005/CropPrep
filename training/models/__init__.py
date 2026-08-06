"""CropFusion multimodal neural architecture (Phase 5).

Implements the complete AI framework consumed by the Phase 6 training loop:

* :class:`TabTransformer` — tabular branch (categorical + continuous tokens).
* :class:`NdviEncoder` / :class:`EviEncoder` — timm backbone image encoders.
* :class:`ImageFusion` — per-timestep NDVI/EVI fusion (concat / weighted sum /
  learnable / attention).
* :class:`TemporalTransformer` — variable-length temporal sequence encoder.
* :class:`CrossAttention` — image attends to tabular (Q=image, K=V=tabular).
* :class:`AdaptiveGatedFusion` — per-sample image / tabular / fusion gates.
* :class:`CrossModalFusionEngine` — cross attention + gated fusion + shared
  encoder as one swappable unit.
* :class:`SharedMultimodalEncoder` — the shared latent space.
* :class:`CropHead` / :class:`YieldHead` / :class:`MultiTaskHeads` — task heads.
* :class:`CropFusionModel` — the full architecture (forward / summary / export).
* :class:`ModelFactory` — construction, architecture registry, runtime, freezing.
* :class:`CheckpointManager` — save / load / resume / partial loading.
* :class:`ModelExporter` — TorchScript / ONNX / future TensorRT.
* Runtime helpers (``runtime.py``) — precision / device / compile / parallelism.
* Loss interfaces (``losses.py``) — never used for training in this phase.

The model consumes exactly the Phase 4 batch dict::

    {
      "tabular":       [B, F],
      "ndvi":          [B, T, 1, H, W],
      "evi":           [B, T, 1, H, W],
      "temporal_mask": [B, T],
    }
"""

from __future__ import annotations

from .adaptive_gate import AdaptiveGatedFusion
from .backbone import TimmImageEncoder
from .checkpoint import CheckpointManager, LoadReport, ResumeState
from .config import (
    CheckpointConfig,
    CropHeadConfig,
    CrossAttentionConfig,
    ExportConfig,
    FusionConfig,
    GatedFusionConfig,
    HeadsConfig,
    ImageEncoderConfig,
    ImageFusionConfig,
    LossConfig,
    ModelConfig,
    RuntimeConfig,
    SharedEncoderConfig,
    TabularModelConfig,
    TemporalModelConfig,
    YieldHeadConfig,
    load_model_config,
    save_model_template,
)
from .cropfusion import CropFusionModel, CropFusionOutput
from .cross_attention import CrossAttention
from .evi_encoder import EviEncoder
from .exceptions import (
    CheckpointError,
    ExportError,
    MissingDependencyError,
    ModelConfigurationError,
    ModelError,
    ModelInputError,
    ShapeMismatchError,
)
from .exporter import ModelExporter
from .factory import ModelFactory
from .fusion_engine import CrossModalFusionEngine, FusionOutput
from .image_fusion import ImageFusion
from .losses import (
    CrossEntropyLoss,
    FocalLoss,
    HuberLoss,
    LabelSmoothingLoss,
    MSELoss,
    WeightedMultiTaskLoss,
)
from .multitask_heads import CropHead, MultiTaskHeads, YieldHead
from .ndvi_encoder import NdviEncoder
from .runtime import (
    amp_context,
    apply_precision,
    apply_runtime,
    compile_model,
    dtype_from_precision,
    enable_gradient_checkpointing,
    move_to_device,
    precision_from_dtype,
    resolve_device,
    wrap_data_parallel,
    wrap_distributed,
)
from .shared_encoder import SharedMultimodalEncoder
from .tabtransformer import TabTransformer
from .temporal_transformer import TemporalTransformer
from .utils import (
    architecture_report,
    build_positional_encoding,
    count_parameters,
    estimate_activation_memory,
    estimate_parameter_memory,
    get_activation,
    layer_summary,
    model_summary,
    parameter_summary,
)
from .validators import expected_batch_shapes, validate_batch

__version__ = "0.1.0"

__all__ = [
    # Architecture
    "CropFusionModel",
    "CropFusionOutput",
    "TabTransformer",
    "NdviEncoder",
    "EviEncoder",
    "TimmImageEncoder",
    "ImageFusion",
    "TemporalTransformer",
    "CrossAttention",
    "AdaptiveGatedFusion",
    "CrossModalFusionEngine",
    "FusionOutput",
    "SharedMultimodalEncoder",
    "CropHead",
    "YieldHead",
    "MultiTaskHeads",
    # Config
    "ModelConfig",
    "TabularModelConfig",
    "ImageEncoderConfig",
    "ImageFusionConfig",
    "TemporalModelConfig",
    "CrossAttentionConfig",
    "GatedFusionConfig",
    "FusionConfig",
    "SharedEncoderConfig",
    "CropHeadConfig",
    "YieldHeadConfig",
    "HeadsConfig",
    "LossConfig",
    "CheckpointConfig",
    "ExportConfig",
    "RuntimeConfig",
    "load_model_config",
    "save_model_template",
    # Factory / persistence / export
    "ModelFactory",
    "CheckpointManager",
    "LoadReport",
    "ResumeState",
    "ModelExporter",
    # Runtime helpers
    "resolve_device",
    "dtype_from_precision",
    "precision_from_dtype",
    "amp_context",
    "apply_precision",
    "move_to_device",
    "compile_model",
    "enable_gradient_checkpointing",
    "wrap_data_parallel",
    "wrap_distributed",
    "apply_runtime",
    # Losses (interfaces — no training in this phase)
    "CrossEntropyLoss",
    "LabelSmoothingLoss",
    "FocalLoss",
    "MSELoss",
    "HuberLoss",
    "WeightedMultiTaskLoss",
    # Validation / summary helpers
    "validate_batch",
    "expected_batch_shapes",
    "count_parameters",
    "parameter_summary",
    "layer_summary",
    "architecture_report",
    "model_summary",
    "estimate_parameter_memory",
    "estimate_activation_memory",
    "get_activation",
    "build_positional_encoding",
    # Exceptions
    "ModelError",
    "ModelConfigurationError",
    "ModelInputError",
    "ShapeMismatchError",
    "MissingDependencyError",
    "CheckpointError",
    "ExportError",
]
