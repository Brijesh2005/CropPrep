"""Model registry — loads, versions, warms up and probes the AI model.

The model is loaded once at startup from a Phase 5/6 checkpoint (or config),
versioned, and warmed up. The registry reports readiness and exposes the model
to the inference engine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from training.models import ModelFactory
from training.models.exceptions import ModelError

from app.core.config import ModelSettings
from app.core.exceptions import InferenceError
from app.core.logging import get_logger, PerformanceTimer

logger = get_logger("model-registry")


class ModelRegistry:
    """Holds the loaded :class:`CropFusionModel` and its metadata."""

    def __init__(self, settings: ModelSettings) -> None:
        self.settings = settings
        self.device = torch.device(
            "cuda"
            if settings.device == "auto" and torch.cuda.is_available()
            else "cpu"
        )
        self.model: Any | None = None
        self.model_config: Any | None = None
        self.version: str = "unloaded"
        self.ready: bool = False
        self._error: str | None = None

    # ------------------------------------------------------------------ #
    # Load / warmup
    # ------------------------------------------------------------------ #

    def load(self) -> None:
        """Load the model from the configured checkpoint / config file."""
        with PerformanceTimer("model.load"):
            if self.settings.checkpoint_path and Path(self.settings.checkpoint_path).exists():
                self.model = ModelFactory.from_checkpoint(self.settings.checkpoint_path)
            elif self.settings.model_config_path and Path(self.settings.model_config_path).exists():
                self.model = ModelFactory.from_config_file(self.settings.model_config_path)
            else:
                raise InferenceError(
                    "no trained model configured (set model.checkpoint_path or "
                    "model.model_config_path)"
                )
        self.model.eval()
        self.model.to(self.device)
        self.model_config = getattr(self.model, "config", None)
        self.version = (
            f"{self.model_config.version}-{self.model_config.name}"
            if self.model_config is not None
            else "unknown"
        )
        self.ready = True
        logger.info("model loaded", version=self.version, device=str(self.device))

    def warmup(self) -> None:
        """Run a single forward to warm the engine and pre-allocate buffers."""
        if self.model is None:
            return
        try:
            batch = self.model.sample_batch(batch_size=self.settings.batch_size)
            with torch.no_grad():
                self.model(batch)
            logger.info("model warmup complete")
        except Exception as exc:
            logger.warning("model warmup failed ({})", exc)

    # ------------------------------------------------------------------ #
    # Probes
    # ------------------------------------------------------------------ #

    def is_ready(self) -> bool:
        return self.ready and self.model is not None

    def version_info(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "device": str(self.device),
            "ready": self.is_ready(),
            "error": self._error,
            "parameters": (
                int(getattr(self.model, "total_parameters", 0)) if self.model else 0
            ),
        }

    # ------------------------------------------------------------------ #
    # Fallback
    # ------------------------------------------------------------------ #

    def fallback_prediction(self, lon: float, lat: float) -> dict[str, Any]:
        """Heuristic fallback when the model is unavailable (opt-in)."""
        import math

        # A deterministic, bounded heuristic so clients still get a response.
        seed = int(abs(lon * 1000 + lat * 1000))
        crops = ["Rice", "Wheat", "Maize", "Coconut"]
        crop = crops[seed % len(crops)]
        return {
            "recommended_crop": crop,
            "crop_probs": {c: round(1.0 / len(crops), 4) for c in crops},
            "expected_yield": round(1.0 + (seed % 50) / 10.0, 2),
            "confidence": 0.0,
            "model_version": "fallback",
            "fallback": True,
        }
