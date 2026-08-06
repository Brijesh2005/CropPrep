"""Model loader (Phase R6).

:class:`ModelLoader` loads a model from a release package using only exported
artefacts. It supports three backends:

* **pytorch** — the self-describing ``model/cropfusion.pt`` is rebuilt via the
  Phase R5 loader (config + state dict, no training code);
* **torchscript** — a traced ``cropfusion.torchscript.pt`` module;
* **onnx** — an ``onnxruntime`` session over ``cropfusion.onnx``.

The loader also exposes the model configuration and metadata, the model
version, warm-up (a forward pass over a deterministic dummy batch derived from
the model config) and a health snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .config import ModelLoadConfig, RuntimeConfig
from .exceptions import ModelLoadError, ModelWarmupError
from .layout import FORMAT_FILES, ReleaseLayout

BACKENDS = ("auto", "pytorch", "torchscript", "onnx")


@dataclass
class ModelHealth:
    """Snapshot of the loaded model."""

    loaded: bool
    backend: str | None = None
    model_version: str | None = None
    config_loaded: bool = False
    metadata_loaded: bool = False
    warmup_ok: bool = False
    parameter_count: int | None = None
    device: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "loaded": self.loaded,
            "backend": self.backend,
            "model_version": self.model_version,
            "config_loaded": self.config_loaded,
            "metadata_loaded": self.metadata_loaded,
            "warmup_ok": self.warmup_ok,
            "parameter_count": self.parameter_count,
            "device": self.device,
        }


class ModelLoader:
    """Load a model from a :class:`ReleaseLayout`.

    Args:
        layout: The release package being loaded.
        config: Validated :class:`RuntimeConfig` (``None`` = defaults). The
            model section controls the backend / device / precision.
    """

    def __init__(
        self, layout: ReleaseLayout, config: RuntimeConfig | None = None
    ) -> None:
        self.layout = layout
        self.config = config or RuntimeConfig()
        self.model_cfg: ModelLoadConfig = self.config.model
        self._backend: str | None = None
        self._model: Any = None
        self._model_metadata: dict[str, Any] = {}
        self._model_config: Any = None
        self._parameter_count: int | None = None
        self._warmup_ok = False

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #

    def load(self, backend: str | None = None) -> Any:
        """Load the model with the given (or resolved) backend.

        Returns:
            The loaded model object (an ``nn.Module`` for pytorch /
            torchscript, an ``onnxruntime.InferenceSession`` for onnx).

        Raises:
            ModelLoadError: When the model cannot be loaded.
        """
        backend = backend or self.model_cfg.backend
        if backend not in BACKENDS:
            raise ModelLoadError(
                f"unknown model backend {backend!r}",
                detail={"supported": list(BACKENDS)},
            )
        resolved = self._resolve_backend(backend)
        self._backend = resolved
        if resolved == "pytorch":
            self._model = self._load_pytorch()
        elif resolved == "torchscript":
            self._model = self._load_torchscript()
        elif resolved == "onnx":
            self._model = self._load_onnx()
        self._load_model_config()
        self._load_model_metadata()
        self._count_parameters()
        return self._model

    def _resolve_backend(self, backend: str) -> str:
        if backend != "auto":
            rel = FORMAT_FILES.get(backend)
            if rel is not None and not self.layout.exists(rel):
                raise ModelLoadError(
                    f"backend {backend!r} requested but {rel} is not in the release",
                    detail={"release": str(self.layout.root)},
                )
            return backend
        for candidate in ("pytorch", "torchscript", "onnx"):
            if self.layout.has_format(candidate):
                return candidate
        raise ModelLoadError(
            "release contains no supported model format "
            "(model/cropfusion.pt, .torchscript.pt or .onnx)",
            detail={"release": str(self.layout.root)},
        )

    def _device(self) -> str:
        device = self.model_cfg.device or self.config.general.device
        if device == "auto":
            return "cuda" if _cuda_available() else "cpu"
        return device

    # -- Backend implementations ------------------------------------------ #

    def _load_pytorch(self) -> Any:
        path = self.layout.artifact(FORMAT_FILES["pytorch"])
        try:
            from training.inference import load_pytorch_model

            model, metadata = load_pytorch_model(path)
        except Exception as exc:  # noqa: BLE001 - surface the load failure
            raise ModelLoadError(
                "failed to restore the pytorch model", detail=str(path)
            ) from exc
        device = self._device()
        try:
            import torch

            model.to(device)
            precision = self.model_cfg.precision
            if precision == "float16":
                model.half()
            elif precision == "bfloat16":
                model.bfloat16()
            model.eval()
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(
                f"failed to move the model to {device} / {precision}",
                detail=str(path),
            ) from exc
        self._model_metadata = dict(metadata or {})
        self._parameter_count = self._model_metadata.get("parameter_count")
        return model

    def _load_torchscript(self) -> Any:
        path = self.layout.artifact(FORMAT_FILES["torchscript"])
        try:
            import torch

            return torch.jit.load(str(path), map_location=self._device())
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(
                "failed to load the TorchScript module", detail=str(path)
            ) from exc

    def _load_onnx(self) -> Any:
        path = self.layout.artifact(FORMAT_FILES["onnx"])
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - guarded by validation
            raise ModelLoadError(
                "onnx backend requires the 'onnxruntime' package",
                detail=str(path),
            ) from exc
        try:
            return ort.InferenceSession(
                str(path), providers=list(self.model_cfg.onnx_providers)
            )
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(
                "failed to create the ONNX inference session", detail=str(path)
            ) from exc

    # -- Config / metadata ------------------------------------------------- #

    def load_config(self) -> Any:
        """Parse and return the release model config (``training.models.ModelConfig``)."""
        self._load_model_config()
        return self._model_config

    def _load_model_config(self) -> None:
        if self._model_config is not None:
            return
        path = self.layout.artifact("configs/model_config.yaml")
        if not path.exists():
            raise ModelLoadError(
                "configs/model_config.yaml is missing", detail=str(self.layout.root)
            )
        try:
            import yaml

            from training.models import ModelConfig

            self._model_config = ModelConfig.model_validate(
                yaml.safe_load(path.read_text(encoding="utf-8"))
            )
        except Exception as exc:  # yaml / pydantic errors
            raise ModelLoadError(
                "failed to parse configs/model_config.yaml", detail=str(path)
            ) from exc

    def load_metadata(self) -> dict[str, Any]:
        """Return the release model metadata (manifest + ``model_metadata.json``)."""
        self._load_model_metadata()
        return dict(self._model_metadata)

    def _load_model_metadata(self) -> None:
        manifest = None
        if self.layout.exists("version/manifest.json"):
            try:
                manifest = self.layout.manifest()
            except Exception:  # ReleaseLayoutError
                manifest = None
        metadata_path = self.layout.artifact("model/model_metadata.json")
        payload: dict[str, Any] = {}
        if metadata_path.exists():
            try:
                import json

                raw = json.loads(metadata_path.read_text(encoding="utf-8"))
                payload = raw if isinstance(raw, dict) else {}
            except Exception:  # noqa: BLE001 - fall through to manifest
                payload = {}
        merged = dict(self._model_metadata)
        merged.update(payload)
        merged["model_version"] = (
            self._model_metadata.get("model_version")
            or payload.get("model_version")
            or (manifest.model_version if manifest else None)
        )
        merged["model_fingerprint"] = (
            self._model_metadata.get("model_fingerprint")
            or payload.get("model_fingerprint")
            or (manifest.model_fingerprint if manifest else "")
        )
        merged["formats"] = (
            payload.get("formats")
            or self._model_metadata.get("formats")
            or self.layout.formats
        )
        self._model_metadata = merged

    # -- Version / counts --------------------------------------------------- #

    def version(self) -> str | None:
        """The model version recorded in the release."""
        return self._model_metadata.get("model_version")

    def _count_parameters(self) -> None:
        if self._parameter_count is not None:
            return
        if self._backend != "pytorch" or self._model is None:
            return
        try:
            self._parameter_count = sum(p.numel() for p in self._model.parameters())
        except Exception:  # pragma: no cover - defensive
            self._parameter_count = None

    # ------------------------------------------------------------------ #
    # Warm-up
    # ------------------------------------------------------------------ #

    def warmup(
        self, steps: int | None = None, batch_size: int | None = None
    ) -> bool:
        """Run ``steps`` forward passes over a config-derived dummy batch.

        Args:
            steps: Number of warm-up iterations (defaults to
                ``model.warmup_steps``).
            batch_size: Batch size for the dummy batch (defaults to
                ``model.warmup_batch_size``).

        Raises:
            ModelWarmupError: When the model is not loaded or warm-up fails.
        """
        if self._model is None:
            raise ModelWarmupError("model is not loaded (call load() first)")
        if self._model_config is None:
            self._load_model_config()
        steps = self.model_cfg.warmup_steps if steps is None else steps
        if steps <= 0:
            self._warmup_ok = True
            return True
        batch_size = self.model_cfg.warmup_batch_size if batch_size is None else batch_size
        try:
            if self._backend == "onnx":
                self._warmup_onnx(batch_size, steps)
            elif self._backend == "torchscript":
                self._warmup_torchscript(batch_size, steps)
            else:
                self._warmup_pytorch(batch_size, steps)
        except ModelWarmupError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ModelWarmupError(
                f"warm-up failed on backend {self._backend!r}: {exc}",
                detail={"steps": steps, "batch_size": batch_size},
            ) from exc
        self._warmup_ok = True
        return True

    def _warmup_pytorch(self, batch_size: int, steps: int) -> None:
        import torch

        batch = build_warmup_batch(self._model_config, batch_size, self._device())
        batch = {
            key: value.to(self._device()) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        with torch.no_grad():
            for _ in range(steps):
                self._model(batch)

    def _warmup_torchscript(self, batch_size: int, steps: int) -> None:
        import torch

        tensors = torchscript_inputs(self._model_config, batch_size, self._device())
        with torch.no_grad():
            for _ in range(steps):
                self._model(*tensors)

    def _warmup_onnx(self, batch_size: int, steps: int) -> None:
        import numpy as np

        cfg = self._model_config
        feeds: dict[str, Any] = {}
        for inp in self._model.get_inputs():
            name = inp.name
            if name == "tabular":
                feeds[name] = np.zeros(
                    (batch_size, cfg.tabular_feature_dim), dtype=np.float32
                )
            elif name in ("ndvi", "evi"):
                seq_len = getattr(cfg.temporal, "max_len", 2) or 2
                size = cfg.image_encoder.input_size or 32
                feeds[name] = np.zeros(
                    (batch_size, seq_len, 1, size, size), dtype=np.float32
                )
            elif name == "temporal_mask":
                seq_len = getattr(cfg.temporal, "max_len", 2) or 2
                feeds[name] = np.ones((batch_size, seq_len), dtype=np.float32)
            else:
                shape = [
                    batch_size if isinstance(d, str) or d is None else int(d)
                    for d in inp.shape
                ]
                feeds[name] = np.zeros(shape, dtype=np.float32)
        for _ in range(steps):
            self._model.run(None, feeds)

    # ------------------------------------------------------------------ #
    # Health
    # ------------------------------------------------------------------ #

    def health(self) -> ModelHealth:
        """A :class:`ModelHealth` snapshot without raising."""
        loaded = self._model is not None
        return ModelHealth(
            loaded=loaded,
            backend=self._backend,
            model_version=self.version(),
            config_loaded=self._model_config is not None,
            metadata_loaded=bool(self._model_metadata),
            warmup_ok=self._warmup_ok,
            parameter_count=self._parameter_count,
            device=self._device() if loaded else None,
        )

    def unload(self) -> None:
        """Release the loaded model / session."""
        if self._backend == "onnx" and self._model is not None:
            try:
                self._model._sess_options = None  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover - defensive
                pass
        self._model = None
        self._backend = None
        self._warmup_ok = False


def build_warmup_batch(cfg: Any, batch_size: int = 2, device: str = "cpu") -> dict[str, Any]:
    """A deterministic zero / one batch derived purely from the model config."""
    import torch

    batch: dict[str, Any] = {
        "tabular": torch.zeros(batch_size, cfg.tabular_feature_dim, dtype=torch.float32),
    }
    if cfg.uses_image:
        seq_len = getattr(cfg.temporal, "max_len", 2) or 2
        size = cfg.image_encoder.input_size or 32
        batch["ndvi"] = torch.zeros(batch_size, seq_len, 1, size, size)
        batch["evi"] = torch.zeros(batch_size, seq_len, 1, size, size)
        batch["temporal_mask"] = torch.ones(batch_size, seq_len, dtype=torch.bool)
    return batch


def torchscript_inputs(cfg: Any, batch_size: int = 2, device: str = "cpu") -> tuple[Any, ...]:
    """Positional tensor inputs matching the traced TorchScript module.

    Order mirrors ``training.models.exporter._ExportWrapper``:
    tabular, ndvi, evi, temporal_mask (subset used by the model).
    """
    import torch

    args: list[Any] = []
    uses_tabular = bool(getattr(cfg, "uses_tabular", True))
    uses_image = bool(getattr(cfg, "uses_image", False))
    if uses_tabular:
        args.append(torch.zeros(batch_size, cfg.tabular_feature_dim, dtype=torch.float32))
    if uses_image:
        seq_len = getattr(cfg.temporal, "max_len", 2) or 2
        size = cfg.image_encoder.input_size or 32
        args.append(torch.zeros(batch_size, seq_len, 1, size, size))
        args.append(torch.zeros(batch_size, seq_len, 1, size, size))
        args.append(torch.ones(batch_size, seq_len, dtype=torch.bool))
    return tuple(args)


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - torch missing
        return False
