"""CropFusionModel — the complete multimodal architecture.

Assembles the Phase 5 architecture exactly as specified::

    tabular -> TabTransformer ------------> tabular embedding
                                             \\            |
    ndvi [B,T,1,H,W] -> NDVI encoder -------> image fusion -> temporal transformer
    evi  [B,T,1,H,W] -> EVI encoder -------/     |             |
                                                    v             v
                                              image embedding    |
                                                    |             |
                                  cross attention (Q=image,K=V=tabular)
                                                    |
                                              adaptive gated fusion
                                                    |
                                              shared multimodal encoder
                                                    |
                                       +------------+------------+
                                       |                         |
                                  crop head                yield head

The model consumes exactly the Phase 4 batch dict produced by
:func:`ai.preprocessing.dataloader.collate_samples`.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from . import runtime as _runtime
from .adaptive_gate import AdaptiveGatedFusion
from .config import ModelConfig
from .cross_attention import CrossAttention
from .evi_encoder import EviEncoder
from .exceptions import ModelConfigurationError
from .fusion_engine import CrossModalFusionEngine
from .image_fusion import ImageFusion
from .multitask_heads import CropHead, MultiTaskHeads, YieldHead
from .ndvi_encoder import NdviEncoder
from .shared_encoder import SharedMultimodalEncoder
from .tabtransformer import TabTransformer
from .temporal_transformer import TemporalTransformer
from .utils import (
    architecture_report,
    count_parameters,
    estimate_activation_memory,
    estimate_parameter_memory,
    layer_summary,
    masked_mean,
    parameter_summary,
    resolve_backbone_name,
)
from .validators import validate_batch, validate_model_config


@dataclass
class CropFusionOutput:
    """Typed forward-pass result of :class:`CropFusionModel`."""

    crop_logits: torch.Tensor | None = None
    yield_pred: torch.Tensor | None = None
    shared_representation: torch.Tensor | None = None
    tabular_embedding: torch.Tensor | None = None
    image_embedding: torch.Tensor | None = None
    #: Raw temporal stream (masked-mean of the fused sequence, projected to
    #: the image-embedding width) when ``fusion.use_temporal_stream``.
    temporal_embedding: torch.Tensor | None = None
    #: Per-sample modality gates from the adaptive fusion (explainability).
    gates: dict[str, torch.Tensor] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "crop_logits": self.crop_logits,
            "yield_pred": self.yield_pred,
            "shared_representation": self.shared_representation,
            "tabular_embedding": self.tabular_embedding,
            "image_embedding": self.image_embedding,
            "temporal_embedding": self.temporal_embedding,
            "gates": self.gates,
        }


class CropFusionModel(nn.Module):
    """Full multimodal CropFusion architecture.

    Args:
        config: Validated :class:`ModelConfig`.

    Attributes:
        output_names: Names of the enabled task heads (``crop`` / ``yield``).
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        validate_model_config(config)
        self.config = config

        self.use_tabular = config.uses_tabular
        self.use_image = config.uses_image
        if not self.use_tabular and not self.use_image:
            raise ModelConfigurationError(
                "ModelConfig requires at least one modality"
            )

        # -- Tabular branch ------------------------------------------------- #
        self.tab_encoder: TabTransformer | None = None
        self.tab_dim = 0
        if self.use_tabular:
            self.tab_encoder = TabTransformer(config.tabular)
            self.tab_dim = self.tab_encoder.output_dim

        # -- Image branch --------------------------------------------------- #
        self.ndvi_encoder: NdviEncoder | None = None
        self.evi_encoder: EviEncoder | None = None
        self.image_fusion: ImageFusion | None = None
        self.temporal_transformer: TemporalTransformer | None = None
        self.image_dim = 0
        if self.use_image:
            image_cfg = config.image_encoder
            if image_cfg.enable_ndvi:
                self.ndvi_encoder = NdviEncoder(
                    backbone=resolve_backbone_name(image_cfg.backbone, image_cfg.ndvi_backbone),
                    pretrained=image_cfg.pretrained,
                    input_size=image_cfg.input_size,
                    channel_expansion=image_cfg.channel_expansion,
                    freeze_backbone=image_cfg.freeze_backbone,
                    drop_path_rate=image_cfg.drop_path_rate,
                )
            if image_cfg.enable_evi:
                self.evi_encoder = EviEncoder(
                    backbone=resolve_backbone_name(image_cfg.backbone, image_cfg.evi_backbone),
                    pretrained=image_cfg.pretrained,
                    input_size=image_cfg.input_size,
                    channel_expansion=image_cfg.channel_expansion,
                    freeze_backbone=image_cfg.freeze_backbone,
                    drop_path_rate=image_cfg.drop_path_rate,
                )
            fusion_cfg = config.image_fusion
            self.image_fusion = ImageFusion(
                ndvi_dim=self.ndvi_encoder.feature_dim if self.ndvi_encoder else 0,
                evi_dim=self.evi_encoder.feature_dim if self.evi_encoder else 0,
                method=fusion_cfg.method,
                hidden_dim=fusion_cfg.hidden_dim,
                dropout=fusion_cfg.dropout,
                activation=config.temporal.activation,
            )
            self.temporal_transformer = TemporalTransformer(
                config.temporal, input_dim=self.image_fusion.out_dim
            )
            self.image_dim = self.temporal_transformer.output_dim

        # -- Cross-modal pathway -------------------------------------------- #
        # Multimodal models route through the CrossModalFusionEngine (cross
        # attention + gated fusion + shared encoder as one unit). Single
        # modality models use a plain shared encoder over their one stream.
        self.fusion_engine: CrossModalFusionEngine | None = None
        self.shared_encoder_standalone: SharedMultimodalEncoder | None = None
        self.temporal_proj: nn.Linear | None = None
        if self.use_tabular and self.use_image:
            self.fusion_engine = CrossModalFusionEngine(
                config, image_dim=self.image_dim, tabular_dim=self.tab_dim
            )
            if config.fusion.use_temporal_stream:
                # Project the pooled per-timestep fused features to the
                # image-embedding width so the engine can gate it as a fourth
                # stream.
                self.temporal_proj = nn.Linear(
                    self.image_fusion.out_dim, self.image_dim
                )
        elif self.use_tabular:
            self.shared_encoder_standalone = SharedMultimodalEncoder(
                self.tab_dim, config.shared_encoder
            )
        else:  # image only
            self.shared_encoder_standalone = SharedMultimodalEncoder(
                self.image_dim, config.shared_encoder
            )
        shared_dim = self.shared_encoder.output_dim

        # -- Task heads ------------------------------------------------------- #
        self.heads = MultiTaskHeads()
        if config.crop_enabled:
            crop_cfg = config.heads.crop
            self.heads.add_head(
                "crop",
                CropHead(
                    in_dim=shared_dim,
                    num_classes=crop_cfg.num_classes,
                    hidden_dim=crop_cfg.hidden_dim,
                    dropout=crop_cfg.dropout,
                    activation=crop_cfg.activation,
                ),
            )
        if config.yield_enabled:
            yield_cfg = config.heads.yield_prediction
            self.heads.add_head(
                "yield",
                YieldHead(
                    in_dim=shared_dim,
                    hidden_dim=yield_cfg.hidden_dim,
                    dropout=yield_cfg.dropout,
                    activation=yield_cfg.activation,
                    output_clamp_min=yield_cfg.output_clamp_min,
                ),
            )
        self.output_names = list(self.heads.names)

    # ------------------------------------------------------------------ #
    # Component access
    # ------------------------------------------------------------------ #

    @property
    def shared_encoder(self) -> SharedMultimodalEncoder:
        """The shared multimodal encoder (engine-owned or standalone).

        Multimodal models keep their shared encoder inside the
        :class:`CrossModalFusionEngine`; single-modality models own it
        directly. This property hides that detail so heads / callers always
        reach the shared encoder the same way.
        """
        if self.fusion_engine is not None:
            return self.fusion_engine.shared_encoder
        assert self.shared_encoder_standalone is not None
        return self.shared_encoder_standalone

    @property
    def cross_attention(self) -> CrossAttention | None:
        """The cross-attention block (engine-owned, ``None`` when disabled /
        single modality)."""
        return self.fusion_engine.cross_attention if self.fusion_engine else None

    @property
    def gated_fusion(self) -> AdaptiveGatedFusion | None:
        """The adaptive gated fusion (engine-owned, ``None`` when disabled /
        single modality)."""
        return self.fusion_engine.gated_fusion if self.fusion_engine else None

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(self, batch: Mapping[str, Any]) -> CropFusionOutput:  # type: ignore[override]
        """Run the full multimodal forward pass.

        Args:
            batch: Phase 4 batch dict with ``tabular`` ``[B, F]``,
                ``ndvi`` / ``evi`` ``[B, T, 1, H, W]`` and ``temporal_mask``
                ``[B, T]``.

        Returns:
            :class:`CropFusionOutput`.
        """
        if self.config.validate_inputs:
            validate_batch(batch, self.config)

        tabular_embedding: torch.Tensor | None = None
        if self.use_tabular:
            tabular_embedding = self.tab_encoder(batch["tabular"])

        image_embedding: torch.Tensor | None = None
        fused_sequence: torch.Tensor | None = None
        if self.use_image:
            ndvi_features = (
                self.ndvi_encoder(batch["ndvi"]) if self.ndvi_encoder is not None else None
            )
            evi_features = (
                self.evi_encoder(batch["evi"]) if self.evi_encoder is not None else None
            )
            fused_sequence = self.image_fusion(ndvi_features, evi_features)
            mask = batch.get("temporal_mask")
            image_embedding = self.temporal_transformer(fused_sequence, mask=mask)

        temporal_embedding: torch.Tensor | None = None
        if (
            self.fusion_engine is not None
            and self.temporal_proj is not None
            and fused_sequence is not None
        ):
            # Raw temporal stream for the gated fusion (fourth gate): the
            # mask-aware mean of the fused per-timestep features, projected to
            # the image-embedding width.
            temporal_embedding = self.temporal_proj(
                masked_mean(fused_sequence, batch.get("temporal_mask"), dim=1)
            )

        gates: dict[str, torch.Tensor] = {}
        if self.fusion_engine is not None:
            engine_out = self.fusion_engine(
                image_embedding,
                tabular_embedding,
                temporal_embedding=temporal_embedding,
            )
            shared = engine_out.shared_embedding
            gates = dict(engine_out.gates)
        elif self.use_tabular:
            shared = self.shared_encoder_standalone(tabular_embedding)
        else:
            shared = self.shared_encoder_standalone(image_embedding)

        head_outputs = self.heads(shared)
        crop_logits = head_outputs.get("crop")
        yield_pred = head_outputs.get("yield")

        return CropFusionOutput(
            crop_logits=crop_logits,
            yield_pred=yield_pred,
            shared_representation=shared,
            tabular_embedding=tabular_embedding,
            image_embedding=image_embedding,
            temporal_embedding=temporal_embedding,
            gates=gates,
        )

    # ------------------------------------------------------------------ #
    # Head registry (future tasks plug in here)
    # ------------------------------------------------------------------ #

    def add_head(self, name: str, module: nn.Module) -> "CropFusionModel":
        """Register an extra task head consuming the shared representation."""
        self.heads.add_head(name, module)
        self.output_names = list(self.heads.names)
        return self

    def remove_head(self, name: str) -> "CropFusionModel":
        self.heads.remove_head(name)
        self.output_names = list(self.heads.names)
        return self

    # ------------------------------------------------------------------ #
    # Export / introspection helpers
    # ------------------------------------------------------------------ #

    def sample_batch(
        self,
        batch_size: int = 2,
        seq_len: int | None = None,
        image_size: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Build a random batch matching the configured input contract.

        Useful for export, summaries and smoke tests.
        """
        if self.use_image:
            if self.ndvi_encoder is not None:
                image_size = image_size or self.ndvi_encoder.input_size[0]
            elif self.evi_encoder is not None:
                image_size = image_size or self.evi_encoder.input_size[0]
        image_size = image_size or 0
        seq_len = seq_len or self.config.temporal.max_len
        batch: dict[str, torch.Tensor] = {}
        if self.use_tabular:
            batch["tabular"] = torch.randn(
                batch_size, self.config.tabular_feature_dim
            )
            # realistic ordinal codes for any categorical slots
            if self.config.tabular.categorical_cardinalities:
                offset = self.config.tabular.numeric_dim
                for col, cardinality in enumerate(
                    self.config.tabular.categorical_cardinalities
                ):
                    batch["tabular"][:, offset + col] = (
                        torch.randint(0, cardinality, (batch_size,))
                    )
        if self.use_image:
            if self.ndvi_encoder is not None:
                batch["ndvi"] = torch.randn(
                    batch_size, seq_len, 1, image_size, image_size
                ) * 0.1
            if self.evi_encoder is not None:
                batch["evi"] = torch.randn(
                    batch_size, seq_len, 1, image_size, image_size
                ) * 0.1
            mask = torch.ones(batch_size, seq_len)
            if seq_len > 1:
                mask[:, -1] = 0.0  # simulate one padded observation
            batch["temporal_mask"] = mask
        return batch

    def forward_export(
        self,
        tabular: torch.Tensor | None = None,
        ndvi: torch.Tensor | None = None,
        evi: torch.Tensor | None = None,
        temporal_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Tensor-only forward used by :class:`~ai.models.exporter.ModelExporter`.

        Returns the enabled outputs — ``(crop_logits, yield_pred,
        shared_representation)`` — as a tuple of non-``None`` tensors so
        TorchScript / ONNX tracing works without dicts.
        """
        batch: dict[str, torch.Tensor] = {}
        if self.use_tabular:
            batch["tabular"] = tabular
        if self.use_image:
            if self.ndvi_encoder is not None:
                batch["ndvi"] = ndvi
            if self.evi_encoder is not None:
                batch["evi"] = evi
            if temporal_mask is not None:
                batch["temporal_mask"] = temporal_mask
        out = self(batch)
        outputs: list[torch.Tensor] = []
        for value in (
            out.crop_logits,
            out.yield_pred,
            out.shared_representation,
        ):
            if value is not None:
                outputs.append(value)
        return tuple(outputs)

    # ------------------------------------------------------------------ #
    # Summary / config helpers
    # ------------------------------------------------------------------ #

    def summary(self, sample_batch: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Parameter count, layer summary, memory and shape estimates.

        Args:
            sample_batch: Optional batch (see :meth:`sample_batch`) used to
                estimate activation memory and to trace real input / output
                shapes through every layer.

        Returns:
            A summary dict with ``parameter_summary``, ``layer_summary``,
            ``memory_estimate``, ``architecture_report`` (per-module shapes)
            and — when ``sample_batch`` is given — ``input_shapes`` and
            ``output_shapes``.
        """
        params = parameter_summary(self)
        memory: dict[str, Any] = {
            "parameters_bytes": estimate_parameter_memory(self),
            "parameters_mb": round(estimate_parameter_memory(self) / (1024**2), 4),
        }
        architecture: list[dict[str, Any]] | None = None
        output_shapes: dict[str, list[int] | None] | None = None
        input_shapes: dict[str, list[int]] | None = None
        if sample_batch is not None:
            activation_bytes = estimate_activation_memory(
                self, forward_fn=lambda: self(sample_batch)
            )
            memory["activation_bytes"] = activation_bytes
            memory["activation_mb"] = round(activation_bytes / (1024**2), 4)

            was_training = self.training
            self.eval()
            captured: dict[str, Any] = {}

            def _run() -> None:
                captured["out"] = self(sample_batch)

            try:
                with torch.no_grad():
                    architecture = architecture_report(self, _run)
            finally:
                self.train(mode=was_training)
            out = captured.get("out")
            if out is not None:
                output_shapes = {
                    "crop_logits": (
                        list(out.crop_logits.shape) if out.crop_logits is not None else None
                    ),
                    "yield_pred": (
                        list(out.yield_pred.shape) if out.yield_pred is not None else None
                    ),
                    "shared_representation": list(out.shared_representation.shape),
                }
            input_shapes = {k: list(v.shape) for k, v in sample_batch.items()}
        return {
            "config": self.config.model_dump(),
            "metadata": self.metadata,
            "output_names": self.output_names,
            "parameter_summary": params,
            "parameter_count": params["total"],
            "layer_summary": layer_summary(self),
            "architecture_report": architecture,
            "input_shapes": input_shapes,
            "output_shapes": output_shapes,
            "memory_estimate": memory,
            "trainable_parameters": params["trainable"],
        }

    @property
    def metadata(self) -> dict[str, Any]:
        """Architecture + environment metadata (stored in checkpoints).

        Includes the config name/version, the schema version of the
        architecture, the enabled outputs, per-modality embedding widths and
        the PyTorch / Python versions used to build the model — everything a
        deployer needs to validate that a checkpoint belongs to the same
        architecture.
        """
        return {
            "name": self.config.name,
            "version": self.config.version,
            "architecture_version": self.config.architecture_version,
            "output_names": list(self.output_names),
            "embedding_dims": {"tabular": self.tab_dim, "image": self.image_dim},
            "shared_dim": self.shared_encoder.output_dim,
            "precision": self.config.runtime.precision,
            "gradient_checkpointing": self.config.runtime.gradient_checkpointing,
            "uses_cross_attention": bool(self.cross_attention),
            "uses_gated_fusion": bool(self.gated_fusion),
            "pytorch_version": torch.__version__,
            "python_version": sys.version.split()[0],
        }

    def save_config(self, path: str | Path) -> Path:
        """Persist the model config as YAML."""
        return self.config.save(path)

    @property
    def total_parameters(self) -> int:
        return count_parameters(self)

    # ------------------------------------------------------------------ #
    # Runtime helpers (delegate to :mod:`ai.models.runtime`)
    # ------------------------------------------------------------------ #

    def to_precision(self, precision: str) -> "CropFusionModel":
        """Convert the model to ``float16`` / ``bfloat16`` / ``float32``.

        Normalisation layers stay in float32 for stability. Returns ``self``.
        """
        return _runtime.apply_precision(self, precision)  # type: ignore[return-value]

    def to_device(self, device: str | torch.device) -> "CropFusionModel":
        """Move the model to a device. Returns ``self``."""
        return _runtime.move_to_device(self, device)  # type: ignore[return-value]

    def enable_gradient_checkpointing(self, enabled: bool = True) -> "CropFusionModel":
        """Enable / disable activation checkpointing on the transformer stacks."""
        return _runtime.enable_gradient_checkpointing(self, enabled)  # type: ignore[return-value]

    def compile(self, mode: str | None = None, backend: str | None = None) -> nn.Module:
        """Compile the model with ``torch.compile``.

        Args:
            mode: ``torch.compile`` mode; defaults to
                ``config.runtime.compile_mode``.
            backend: Explicit compile backend (``inductor`` default; ``eager``
                useful for testing).

        Returns:
            The compiled module (a new object wrapping this model).
        """
        return _runtime.compile_model(
            self, mode=mode or self.config.runtime.compile_mode, backend=backend
        )
