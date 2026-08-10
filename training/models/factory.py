"""Model factory — construction, pretrained loading and configuration helpers.

:class:`ModelFactory` centralises every way a :class:`CropFusionModel` is
built, so callers never wire modules by hand:

* :meth:`ModelFactory.create` — build from a :class:`ModelConfig`.
* :meth:`ModelFactory.from_config_file` — build from a YAML file.
* :meth:`ModelFactory.from_preprocessor` — build directly from a fitted
  Phase 4 :class:`~ai.preprocessing.master_pipeline.Preprocessor`.
* :meth:`ModelFactory.from_checkpoint` — rebuild + restore weights from a
  checkpoint.
* Architecture registry + version management — future architectures register
  under a name and are rebuilt from checkpoints by that name.
* Backbone / pretrained loading, layer freezing, runtime configuration and
  config persistence.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from torch import nn

from training.dataset_manager.config import deep_merge

from . import runtime as _runtime
from .checkpoint import CheckpointManager, LoadReport
from .config import ModelConfig, load_model_config
from .cropfusion import CropFusionModel
from .exceptions import ModelConfigurationError
from .utils import freeze_matching

_BACKBONE_PATTERNS = (
    r"^ndvi_encoder\.backbone\.",
    r"^evi_encoder\.backbone\.",
)


class ModelFactory:
    """Construction / loading / freezing helpers for CropFusionModel."""

    #: Registered architectures (name -> model class). ``cropfusion_v1`` is
    #: the built-in; future architectures register via
    #: :meth:`register_architecture` so checkpoints can be rebuilt by name.
    _ARCHITECTURES: dict[str, type[nn.Module]] = {
        "cropfusion_v1": CropFusionModel,
    }

    # ------------------------------------------------------------------ #
    # Architecture registry / version management
    # ------------------------------------------------------------------ #

    @classmethod
    def register_architecture(cls, name: str, model_cls: type[nn.Module]) -> None:
        """Register a model class under an architecture name.

        Args:
            name: Unique architecture name (stored in checkpoints).
            model_cls: An ``nn.Module`` subclass whose constructor accepts a
                :class:`ModelConfig`.

        Raises:
            ModelConfigurationError: When ``model_cls`` is not an ``nn.Module``
                subclass.
        """
        if not isinstance(model_cls, type) or not issubclass(model_cls, nn.Module):
            raise ModelConfigurationError(
                f"registered architecture {name!r} must be an nn.Module subclass",
                detail=type(model_cls).__name__,
            )
        cls._ARCHITECTURES[name] = model_cls

    @classmethod
    def architecture_names(cls) -> list[str]:
        """Names of all registered architectures."""
        return list(cls._ARCHITECTURES)

    @classmethod
    def resolve_architecture(cls, config: Any) -> type[nn.Module]:
        """Resolve the model class for a config.

        Uses ``config.name`` when it matches a registered architecture;
        otherwise falls back to the built-in :class:`CropFusionModel` (so
        arbitrary display names still build).
        """
        name = (
            config.get("name")
            if isinstance(config, Mapping)
            else getattr(config, "name", None)
        )
        return cls._ARCHITECTURES.get(name, CropFusionModel)

    @classmethod
    def _require_architecture(cls, name: str) -> type[nn.Module]:
        if name not in cls._ARCHITECTURES:
            raise ModelConfigurationError(
                f"unknown architecture {name!r}; registered architectures: "
                f"{sorted(cls._ARCHITECTURES)}",
                detail=name,
            )
        return cls._ARCHITECTURES[name]

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    @classmethod
    def create(
        cls,
        config: ModelConfig | Mapping[str, Any],
        *,
        architecture: str | None = None,
    ) -> CropFusionModel:
        """Build a model from a config (or config dict).

        Args:
            config: A :class:`ModelConfig` or plain dict.
            architecture: Explicit architecture name to build; when given it
                must be registered. Defaults to :meth:`resolve_architecture`.
        """
        model_config = config if isinstance(config, ModelConfig) else ModelConfig(**config)
        model_cls = (
            cls._require_architecture(architecture)
            if architecture is not None
            else cls.resolve_architecture(model_config)
        )
        return model_cls(model_config)  # type: ignore[call-arg]

    @classmethod
    def from_config_file(cls, config_path: str | Path) -> CropFusionModel:
        """Load a YAML model config and build the model from it."""
        config = load_model_config(config_path)
        return cls.create(config)

    @classmethod
    def build_config(
        cls, preprocessor: Any, **overrides: Any
    ) -> ModelConfig:
        """Derive a :class:`ModelConfig` from a fitted Phase 4 preprocessor."""
        return ModelConfig.from_preprocessor(preprocessor, **overrides)

    @classmethod
    def from_preprocessor(
        cls,
        preprocessor: Any,
        config_path: str | Path | None = None,
        **overrides: Any,
    ) -> CropFusionModel:
        """Build a model directly from a fitted Phase 4 preprocessor.

        The tabular schema, crop class count and default image size are
        derived automatically from the preprocessor. Precedence (highest
        first):

        ``**overrides`` kwargs > derived-from-preprocessor > ``config_path``
        file > built-in defaults.

        The tabular schema always comes from the preprocessor (a model file
        cannot know it); everything else — backbone, dims, heads, loss — is
        filled from the file/overrides.

        Args:
            preprocessor: Fitted :class:`~ai.preprocessing.master_pipeline.
                Preprocessor`.
            config_path: Optional YAML providing architecture settings
                (schema fields are overridden by the derived values).
            **overrides: :class:`ModelConfig` fields to override.

        Raises:
            ModelConfigurationError: If the preprocessor is not fitted or the
                derived config is invalid.
        """
        if not getattr(preprocessor, "fitted", False):
            raise ModelConfigurationError(
                "preprocessor must be fitted before building a model "
                "(call preprocessor.fit(train_obs, extractor=...))"
            )
        derived = ModelConfig._derived_schema(preprocessor)

        merged: dict[str, Any] = {}
        if config_path is not None:
            merged = load_model_config(config_path).model_dump()
        merged = deep_merge(merged, overrides)

        # Schema fields are owned by the preprocessor.
        merged["tabular"] = {**merged.get("tabular", {}), **derived["tabular"]}
        # Crop class count: derived unless the user chose a nonzero value.
        crop = merged.setdefault("heads", {}).setdefault("crop", {})
        crop["num_classes"] = crop.get("num_classes") or derived["heads"]["crop"]["num_classes"]
        # Input size: derived only when the user left it unset (null).
        if merged.get("image_encoder", {}).get("input_size") is None:
            merged.setdefault("image_encoder", {})["input_size"] = derived["image_encoder"]["input_size"]
        # Temporal capacity: at least enough for the sequence length.
        merged.setdefault("temporal", {})["max_len"] = max(
            merged.get("temporal", {}).get("max_len") or 0,
            derived["temporal"]["max_len"],
        )
        return cls.create(merged)

    @classmethod
    def from_config(
        cls,
        config: ModelConfig | Mapping[str, Any],
        *,
        architecture: str | None = None,
    ) -> CropFusionModel:
        """Build a model from a config (or config dict) — alias of :meth:`create`.

        Provided so the inference release loader can rebuild the architecture
        from ``model.yaml`` on the state-dict fallback path without importing
        the training stack.
        """
        return cls.create(config, architecture=architecture)

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> CropFusionModel:
        """Rebuild a model from a checkpoint (config + weights).

        The checkpoint's ``architecture`` (falling back to its config name) is
        used to resolve the model class, so a registered future architecture is
        rebuilt with its own class, not the built-in one.

        The rebuilt model is created with ``image_encoder.pretrained=False`` —
        the checkpoint already contains the trained backbone weights, so the
        ImageNet state is never re-downloaded (important on offline inference
        / export hosts).

        Args:
            path: Checkpoint file written by
                :class:`~ai.models.checkpoint.CheckpointManager`.

        Returns:
            A :class:`CropFusionModel` with checkpoint weights loaded.
        """
        state = CheckpointManager.load(path)
        model_config = state.get("model_config")
        if not model_config:
            raise ModelConfigurationError(
                "checkpoint does not contain a model config; cannot rebuild "
                "the architecture",
                detail=str(path),
            )
        rebuild_config = deepcopy(model_config)
        image_encoder = (
            rebuild_config.get("image_encoder")
            if isinstance(rebuild_config, Mapping)
            else getattr(rebuild_config, "image_encoder", None)
        )
        if isinstance(image_encoder, Mapping):
            image_encoder["pretrained"] = False
        elif image_encoder is not None:
            image_encoder.pretrained = False
        architecture = state.get("architecture") or (
            model_config.get("name") if isinstance(model_config, Mapping) else None
        )
        model = cls.create(rebuild_config, architecture=architecture)
        report = CheckpointManager.load_state_into(model, path, strict=True)
        return model

    # ------------------------------------------------------------------ #
    # Runtime configuration
    # ------------------------------------------------------------------ #

    @classmethod
    def create_with_runtime(
        cls,
        config: ModelConfig | Mapping[str, Any],
        *,
        architecture: str | None = None,
    ) -> nn.Module:
        """Build a model and apply its ``RuntimeConfig`` in one call."""
        model = cls.create(config, architecture=architecture)
        return _runtime.apply_runtime(model)

    @staticmethod
    def apply_runtime(model: nn.Module, runtime: Any | None = None) -> nn.Module:
        """Apply a :class:`~ai.models.config.RuntimeConfig` to a built model."""
        return _runtime.apply_runtime(model, runtime)

    @staticmethod
    def set_precision(model: nn.Module, precision: str) -> nn.Module:
        """Convert a model to ``float16`` / ``bfloat16`` / ``float32``."""
        return _runtime.apply_precision(model, precision)

    @staticmethod
    def compile(model: nn.Module, mode: str = "default") -> nn.Module:
        """Compile a model with ``torch.compile``."""
        return _runtime.compile_model(model, mode=mode)

    @staticmethod
    def enable_gradient_checkpointing(
        model: nn.Module, enabled: bool = True
    ) -> nn.Module:
        """Enable / disable activation checkpointing on the transformer stacks."""
        return _runtime.enable_gradient_checkpointing(model, enabled)

    # ------------------------------------------------------------------ #
    # Pretrained / backbone loading
    # ------------------------------------------------------------------ #

    @staticmethod
    def load_backbone(
        model: CropFusionModel,
        path: str | Path,
        *,
        include: Sequence[str] | None = None,
        exclude: Sequence[str] | None = None,
    ) -> LoadReport:
        """Load backbone weights into the NDVI/EVI encoders only.

        Args:
            model: A :class:`CropFusionModel` (image branch enabled).
            path: Checkpoint / state-dict file.
            include: Extra regex patterns for keys to load.
            exclude: Regex patterns for keys to skip.

        Returns:
            :class:`LoadReport` describing the partial load.
        """
        patterns = list(_BACKBONE_PATTERNS)
        if include:
            patterns.extend(include)
        return CheckpointManager.partial_load(
            model, path, include=patterns, exclude=list(exclude or [])
        )

    @staticmethod
    def load_pretrained(
        model: CropFusionModel, path: str | Path, *, strict: bool = False
    ) -> LoadReport:
        """Load a full pretrained checkpoint into a model (partial by default)."""
        return CheckpointManager.load_state_into(model, path, strict=strict)

    # ------------------------------------------------------------------ #
    # Freezing
    # ------------------------------------------------------------------ #

    @staticmethod
    def freeze_layers(
        model: CropFusionModel, patterns: Sequence[str]
    ) -> list[str]:
        """Freeze every parameter whose name matches any regex pattern."""
        return freeze_matching(model, patterns)

    @staticmethod
    def freeze_backbone(model: CropFusionModel) -> list[str]:
        """Freeze both timm backbones (NDVI + EVI)."""
        if not model.use_image:
            raise ModelConfigurationError(
                "freeze_backbone requires an image branch (image_encoder.backbone)"
            )
        return freeze_matching(model, _BACKBONE_PATTERNS)

    # ------------------------------------------------------------------ #
    # Config persistence
    # ------------------------------------------------------------------ #

    @staticmethod
    def save_config(config: ModelConfig | Mapping[str, Any], path: str | Path) -> Path:
        """Persist a model config to a YAML file."""
        model_config = config if isinstance(config, ModelConfig) else ModelConfig(**config)
        return model_config.save(path)
