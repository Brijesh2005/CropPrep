"""Model loader port (architecture contract).

Loads an *exported* model artifact set from ``application/inference_package``
/ ``application/models``. The loader is deliberately decoupled from the
Training Platform: it reads versioned artifacts produced by the training
export pipeline (``shared.interfaces.ModelExporter``) and never calls into
``training.models``.

R1.4 does not implement loading. The contract fixes:

- the artifact layout (see ``application/inference_package``),
- a ``ModelPackage`` value object that bundles what a future engine needs,
- readiness / versioning hooks for health probes and warmup.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from shared.versioning import ModelVersion


@dataclass(frozen=True, slots=True)
class ModelPackage:
    """A loaded model artifact bundle (future runtime handle).

    Fields are intentionally ``Any``: the concrete engine will bind them to
    the real model / preprocessor objects in a later phase without changing
    this contract.
    """

    version: ModelVersion
    weights_path: Path
    model_config: dict[str, Any] = field(default_factory=dict)
    preprocessor: Any | None = None
    scaler: Any | None = None
    label_encoder: Any | None = None
    device: str = "cpu"


class ModelLoader(ABC):
    """Port for loading / unloading the exported model artifact set."""

    @abstractmethod
    def load(self, package_dir: Path, *, pinned: str | None = None) -> ModelPackage:
        """Load the model package from ``package_dir`` (pinned version optional).

        ``package_dir`` is typically ``application/inference_package`` and the
        weights live under ``application/models``.
        """

    @abstractmethod
    def unload(self) -> None:
        """Release the loaded model and free device memory."""

    @abstractmethod
    def is_ready(self) -> bool:
        """Return whether a model is currently loaded and usable."""


__all__ = ["ModelLoader", "ModelPackage"]
