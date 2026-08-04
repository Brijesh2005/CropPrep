"""Ports (abstract interfaces) for the preprocessing pipeline.

The pipeline is built from small, composable stages (scalers, encoders,
per-modality pipelines) that all implement a fit/transform contract. The
:class:`Pipeline` interface is the contract for every per-modality pipeline
(Tabular / Image / Temporal / Label); :class:`Preprocessor` is the master
orchestrator consumed by PyTorch datasets and loaders.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class Transformer(ABC):
    """A fitted, persistable numeric transform (scaler or encoder)."""

    fitted: bool

    @abstractmethod
    def fit(self, X: Any) -> "Transformer":
        """Learn transform parameters from ``X``."""

    @abstractmethod
    def transform(self, X: Any) -> Any:
        """Apply the transform to ``X``."""

    @abstractmethod
    def fit_transform(self, X: Any) -> Any:
        """Fit then transform."""

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialisable state (for persistence)."""

    @classmethod
    @abstractmethod
    def from_dict(cls, state: dict[str, Any]) -> "Transformer":
        """Rebuild a transformer from :meth:`to_dict` output."""

    def save(self, path: str | Path) -> Path:
        """Persist via pickle."""
        import pickle

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as fh:
            pickle.dump(self.to_dict(), fh)
        return out

    @classmethod
    def load(cls, path: str | Path) -> "Transformer":
        """Load a persisted transformer."""
        import pickle

        # Transformers are persisted to and loaded from the app's own artifact
        # registry (configured preprocessor dir), never from user uploads.
        with open(path, "rb") as fh:
            state = pickle.load(fh)  # nosec B301
        return cls.from_dict(state)


class Pipeline(ABC):
    """A per-modality preprocessing stage (fit / transform / validate / summary)."""

    @abstractmethod
    def fit(self, samples: Any) -> "Pipeline":
        """Fit internal parameters on the training samples only."""

    @abstractmethod
    def transform(self, sample: Any) -> Any:
        """Transform one sample into the pipeline's tensor output."""

    @abstractmethod
    def validate(self, sample: Any) -> list[Any]:
        """Return a list of validation issues for a sample (empty = valid)."""

    @abstractmethod
    def summary(self) -> dict[str, Any]:
        """Describe fitted state (feature names, sizes, stats)."""

    @abstractmethod
    def save(self, directory: str | Path) -> Path:
        """Persist fitted artifacts under ``directory``."""

    @classmethod
    @abstractmethod
    def load(cls, directory: str | Path) -> "Pipeline":
        """Restore a fitted pipeline from ``directory``."""
