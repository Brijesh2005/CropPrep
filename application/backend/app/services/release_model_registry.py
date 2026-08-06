"""Model registry — loads, versions, warms up and probes the model.

REPLACES ``app/services/model_registry.py`` for the Prediction Platform.
Unlike the R1-R5 registry, this one never imports the Dataset Manager, STAM,
or ``training.models.ModelFactory.from_checkpoint``. It only loads the
exported ``cropfusion_release/`` package via
``inference_package.release.ReleasePackageLoader``.

Point ``model.checkpoint_path`` (in ``app.core.config.ModelSettings``) at the
release directory, e.g.::

    model:
      checkpoint_path: "/srv/cropfusion_release"

(The field is reused rather than adding a new setting, to avoid touching
``app/core/config.py``.)
"""

from __future__ import annotations

from typing import Any

from app.core.config import ModelSettings
from app.core.exceptions import InferenceError
from app.core.logging import PerformanceTimer, get_logger
from inference_package.release import ReleasePackage, ReleasePackageError, ReleasePackageLoader

logger = get_logger("model-registry")


class ModelRegistry:
    """Holds the loaded release package and reports readiness / version info.

    Public interface is intentionally identical to the training-backed
    registry it replaces (``is_ready``, ``version_info``, ``warmup``,
    ``fallback_prediction``, ``.model``, ``.device``, ``.version``) so
    ``HealthService`` and the dependency wiring need no changes.
    """

    def __init__(self, settings: ModelSettings) -> None:
        self.settings = settings
        self.package: ReleasePackage | None = None
        self.model: Any | None = None
        self.version: str = "unloaded"
        self.dataset_version: str = "unloaded"
        self.device: str = "cpu"
        self.ready: bool = False
        self._error: str | None = None

    # ------------------------------------------------------------------ #
    # Load / warmup
    # ------------------------------------------------------------------ #

    def load(self) -> None:
        """Load and validate the release package (checksums, manifest, compat)."""
        package_dir = self.settings.checkpoint_path
        if not package_dir:
            raise InferenceError(
                "no release package configured (set model.checkpoint_path to the "
                "cropfusion_release/ directory)"
            )
        with PerformanceTimer("model.load"):
            try:
                loader = ReleasePackageLoader(package_dir, device=self.settings.device)
                self.package = loader.load()
            except ReleasePackageError as exc:
                self._error = str(exc)
                raise InferenceError(f"release package validation failed: {exc}") from exc

        self.model = self.package.model
        self.version = self.package.model_version
        self.dataset_version = self.package.dataset_version
        self.device = self.package.device
        self.ready = True
        self._error = None
        logger.info(
            "release package loaded",
            version=self.version,
            dataset_version=self.dataset_version,
            device=self.device,
            model_kind=self.package.model_kind,
        )

    def warmup(self) -> None:
        """Run a single dummy forward pass to pre-allocate buffers/kernels."""
        if self.model is None or self.package is None:
            return
        try:
            import torch

            input_dim = int(self.package.model_config.get("input_dim", 0))
            if input_dim <= 0:
                logger.info("warmup skipped: model.yaml has no input_dim")
                return
            dummy = torch.zeros((1, input_dim), device=self.device)
            with torch.no_grad():
                self.model(dummy)
            logger.info("model warmup complete")
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("model warmup failed ({})", exc)

    # ------------------------------------------------------------------ #
    # Probes
    # ------------------------------------------------------------------ #

    def is_ready(self) -> bool:
        return self.ready and self.model is not None

    def version_info(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "dataset_version": self.dataset_version,
            "device": self.device,
            "ready": self.is_ready(),
            "error": self._error,
            "model_kind": self.package.model_kind if self.package else None,
            "metrics": self.package.metrics if self.package else {},
        }

    # ------------------------------------------------------------------ #
    # Fallback
    # ------------------------------------------------------------------ #

    def fallback_prediction(self, lon: float, lat: float) -> dict[str, Any]:
        """Deterministic heuristic used only when the release package failed to load."""
        seed = int(abs(lon * 1000 + lat * 1000))
        crops = ["Rice", "Wheat", "Maize", "Coconut"]
        crop = crops[seed % len(crops)]
        return {
            "recommended_crop": crop,
            "crop_probs": {c: round(1.0 / len(crops), 4) for c in crops},
            "expected_yield": round(1.0 + (seed % 50) / 10.0, 2),
            "confidence": 0.0,
            "model_version": "fallback",
            "dataset_version": "fallback",
            "fallback": True,
        }


__all__ = ["ModelRegistry"]
