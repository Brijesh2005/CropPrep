"""Preprocess loader (Phase R6).

:class:`PreprocessLoader` loads the fitted preprocessing pipelines shipped in
a release package (``feature_scalers.pkl`` and ``label_encoder.pkl``) together
with their configuration and metadata. It never fits anything — the pipelines
are loaded exactly as exported by the training platform.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import PreprocessConfig, RuntimeConfig
from .exceptions import PreprocessLoadError
from .layout import ReleaseLayout


@dataclass
class PreprocessHealth:
    """Snapshot of the loaded preprocessing pipelines."""

    loaded: bool
    feature_names: list[str] = field(default_factory=list)
    num_features: int | None = None
    num_classes: int | None = None
    fitted: bool = False
    config_loaded: bool = False
    metadata_loaded: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "loaded": self.loaded,
            "feature_names": list(self.feature_names),
            "num_features": self.num_features,
            "num_classes": self.num_classes,
            "fitted": self.fitted,
            "config_loaded": self.config_loaded,
            "metadata_loaded": self.metadata_loaded,
        }


class PreprocessLoader:
    """Load ``feature_scalers.pkl`` / ``label_encoder.pkl`` from a release.

    Args:
        layout: The release package being loaded.
        config: Validated :class:`RuntimeConfig` (``None`` = defaults).
    """

    def __init__(
        self, layout: ReleaseLayout, config: RuntimeConfig | None = None
    ) -> None:
        self.layout = layout
        self.config = config or RuntimeConfig()
        self.preprocess_cfg: PreprocessConfig = self.config.preprocess
        self._feature_scalers: Any = None
        self._label_encoder: Any = None
        self._config: dict[str, Any] = {}
        self._metadata: dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #

    def load(self) -> "PreprocessLoader":
        """Load the preprocessing pipelines.

        Raises:
            PreprocessLoadError: When a required pipeline is missing or cannot
                be unpickled.
        """
        scalers_path = self.layout.artifact("preprocess/feature_scalers.pkl")
        encoder_path = self.layout.artifact("preprocess/label_encoder.pkl")
        if self.preprocess_cfg.required:
            for label, path in (("feature_scalers", scalers_path),
                                ("label_encoder", encoder_path)):
                if not path.exists():
                    raise PreprocessLoadError(
                        f"preprocess/{label}.pkl is missing",
                        detail=str(path),
                    )
        self._feature_scalers = _unpickle(scalers_path, "feature_scalers")
        self._label_encoder = _unpickle(encoder_path, "label_encoder")
        self._load_config()
        self._load_metadata()
        return self

    def _load_config(self) -> None:
        path = self.layout.artifact("preprocess/preprocess_metadata.json")
        if path.exists():
            self._config = _load_json(path)
            self._metadata = dict(self._config)

    def load_config(self) -> dict[str, Any]:
        """The preprocessing configuration (``preprocess_metadata.json``)."""
        self._load_config()
        return dict(self._config)

    def load_metadata(self) -> dict[str, Any]:
        """Metadata derived from the loaded pipelines + shipped JSON."""
        self._load_metadata()
        return dict(self._metadata)

    def _load_metadata(self) -> None:
        if self._feature_scalers is None:
            return
        feature_names = list(getattr(self._feature_scalers, "feature_names", None) or [])
        num_classes = int(
            getattr(self._label_encoder, "num_classes", 0) or 0
        ) if self._label_encoder is not None else 0
        self._metadata = {
            "feature_names": feature_names,
            "num_features": len(feature_names) if feature_names else None,
            "num_classes": num_classes,
            "fitted": bool(
                getattr(self._feature_scalers, "fitted", False)
                and getattr(self._label_encoder, "fitted", False)
            ),
        }

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #

    @property
    def feature_scalers(self) -> Any:
        """The fitted tabular pipeline (``feature_scalers.pkl``)."""
        return self._feature_scalers

    @property
    def label_encoder(self) -> Any:
        """The fitted label pipeline (``label_encoder.pkl``)."""
        return self._label_encoder

    @property
    def feature_names(self) -> list[str]:
        return list(getattr(self._feature_scalers, "feature_names", None) or [])

    @property
    def num_classes(self) -> int | None:
        if self._label_encoder is None:
            return None
        value = int(getattr(self._label_encoder, "num_classes", 0) or 0)
        return value or None

    # ------------------------------------------------------------------ #
    # Health
    # ------------------------------------------------------------------ #

    def health(self) -> PreprocessHealth:
        loaded = self._feature_scalers is not None and self._label_encoder is not None
        return PreprocessHealth(
            loaded=loaded,
            feature_names=self.feature_names,
            num_features=len(self.feature_names) if self.feature_names else None,
            num_classes=self.num_classes,
            fitted=bool(self._metadata.get("fitted")),
            config_loaded=bool(self._config),
            metadata_loaded=bool(self._metadata),
        )

    def unload(self) -> None:
        self._feature_scalers = None
        self._label_encoder = None
        self._metadata = {}


def _unpickle(path: Path, name: str) -> Any:
    if not path.exists():
        return None
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except Exception as exc:  # noqa: BLE001 - surface the unpickle failure
        raise PreprocessLoadError(
            f"failed to unpickle {name}", detail=str(path)
        ) from exc


def _load_json(path: Path) -> dict[str, Any]:
    import json

    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}
