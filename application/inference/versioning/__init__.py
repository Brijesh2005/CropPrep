"""Model version resolution port (architecture contract).

Resolves which exported model to serve: a pinned semantic version, ``latest``,
or a fallback layout such as ``cropfusion_v1.pt`` / ``cropfusion_v2.pt`` /
``cropfusion_latest.pt`` under ``application/models``. Uses ``shared`` version
types so the inference platform shares the canonical vocabulary.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from shared.versioning import ModelVersion


class ModelVersionResolver(ABC):
    """Port for resolving the served model version."""

    @abstractmethod
    def resolve(self, pinned: str | None = None) -> ModelVersion:
        """Return the model version to serve (``pinned`` or ``latest``)."""

    @abstractmethod
    def list_available(self) -> list[ModelVersion]:
        """List the exported model versions available in ``application/models``."""


__all__ = ["ModelVersionResolver"]
