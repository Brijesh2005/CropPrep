"""Loads and validates an exported ``cropfusion_release/`` package.

This is the *only* code path the Prediction Platform uses to obtain a model
and its supporting artifacts. It never imports the Dataset Manager, STAM,
Kaggle download code, or the training pipeline. If the release package was
exported with a TorchScript model (recommended), loading is fully
self-contained. If it was exported as a raw ``state_dict`` + ``model.yaml``,
this loader falls back to instantiating the architecture via
``training.models.ModelFactory`` purely to reconstruct the module shape for
``load_state_dict`` — no training-time behaviour (data loading, optimizers,
augmentation) is touched.

Validation order (fails fast, in this order):
    1. directory + required files exist
    2. ``version/manifest.json`` format/schema is understood
    3. every file's sha256 matches ``version/checksum.json``
    4. ``configs/model.yaml`` / ``configs/inference.yaml`` parse as YAML
"""

from __future__ import annotations

import hashlib
import json
import pickle
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml

from inference_package.release.manifest import (
    RELEASE_PACKAGE_FILES,
    SUPPORTED_MANIFEST_SCHEMA_VERSION,
    SUPPORTED_PACKAGE_FORMAT,
)


class ReleasePackageError(RuntimeError):
    """Raised for any missing file, checksum mismatch, or version incompatibility."""


@dataclass(slots=True)
class ReleasePackage:
    """Everything the inference engine needs, loaded from ``cropfusion_release/``."""

    root: Path
    model: Any
    model_kind: str  # "torchscript" | "state_dict"
    scaler: Any
    label_encoder: Any
    model_config: dict[str, Any]
    inference_config: dict[str, Any]
    historical_context: pd.DataFrame
    location_index: pd.DataFrame
    village_metadata: pd.DataFrame
    metadata_db_path: Path
    metrics: dict[str, Any]
    manifest: dict[str, Any]
    model_version: str
    dataset_version: str
    device: str = "cpu"

    def metadata_connection(self) -> sqlite3.Connection:
        """Open a fresh read-only connection to ``metadata.db``."""
        uri = f"file:{self.metadata_db_path}?mode=ro"
        return sqlite3.connect(uri, uri=True)


class ReleasePackageLoader:
    """Validates and loads a ``cropfusion_release/`` directory into memory."""

    def __init__(self, package_dir: str | Path, *, device: str = "auto") -> None:
        self.package_dir = Path(package_dir)
        self.device = self._resolve_device(device)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def load(self) -> ReleasePackage:
        self._validate_files_exist()
        manifest = self._load_json("version/manifest.json")
        self._validate_manifest(manifest)
        checksums = self._load_json("version/checksum.json")
        self._validate_checksums(checksums)

        model_config = self._load_yaml("configs/model.yaml")
        inference_config = self._load_yaml("configs/inference.yaml")
        metrics = self._load_json("reports/metrics.json")

        model, model_kind = self._load_model(model_config)
        scaler = self._load_pickle("preprocess/scaler.pkl")
        label_encoder = self._load_pickle("preprocess/label_encoder.pkl")

        historical_context = pd.read_parquet(self._path("metadata/historical_context.parquet"))
        location_index = pd.read_parquet(self._path("metadata/location_index.parquet"))
        village_metadata = pd.read_parquet(self._path("metadata/village_metadata.parquet"))

        return ReleasePackage(
            root=self.package_dir,
            model=model,
            model_kind=model_kind,
            scaler=scaler,
            label_encoder=label_encoder,
            model_config=model_config,
            inference_config=inference_config,
            historical_context=historical_context,
            location_index=location_index,
            village_metadata=village_metadata,
            metadata_db_path=self._path("metadata/metadata.db"),
            metrics=metrics,
            manifest=manifest,
            model_version=str(manifest.get("model_version", "unknown")),
            dataset_version=str(manifest.get("dataset_version", "unknown")),
            device=self.device,
        )

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def _validate_files_exist(self) -> None:
        if not self.package_dir.is_dir():
            raise ReleasePackageError(f"release package directory not found: {self.package_dir}")
        missing = [
            a.rel_path
            for a in RELEASE_PACKAGE_FILES
            if a.required and not self._path(a.rel_path).exists()
        ]
        if missing:
            raise ReleasePackageError(
                "release package is missing required files: " + ", ".join(missing)
            )

    def _validate_manifest(self, manifest: dict[str, Any]) -> None:
        fmt = manifest.get("format")
        schema_version = manifest.get("schema_version")
        if fmt != SUPPORTED_PACKAGE_FORMAT:
            raise ReleasePackageError(
                f"unrecognised package format {fmt!r}, expected {SUPPORTED_PACKAGE_FORMAT!r}"
            )
        if not isinstance(schema_version, int) or schema_version > SUPPORTED_MANIFEST_SCHEMA_VERSION:
            raise ReleasePackageError(
                f"unsupported manifest schema_version={schema_version!r} "
                f"(this loader supports up to {SUPPORTED_MANIFEST_SCHEMA_VERSION})"
            )

    def _validate_checksums(self, checksums: dict[str, Any]) -> None:
        files: dict[str, str] = checksums.get("files", checksums)
        mismatched: list[str] = []
        for rel_path, expected in files.items():
            full = self._path(rel_path)
            if not full.exists():
                continue  # already caught by _validate_files_exist for required files
            actual = self._sha256(full)
            if actual != expected:
                mismatched.append(rel_path)
        if mismatched:
            raise ReleasePackageError(
                "checksum mismatch for: " + ", ".join(mismatched) +
                " (release package is corrupted or was modified after export)"
            )

    # ------------------------------------------------------------------ #
    # Loaders
    # ------------------------------------------------------------------ #

    def _load_model(self, model_config: dict[str, Any]) -> tuple[Any, str]:
        weights_path = self._path("model/cropfusion.pt")
        try:
            model = torch.jit.load(str(weights_path), map_location=self.device)
            model.eval()
            return model, "torchscript"
        except RuntimeError:
            pass  # not a TorchScript archive — fall back to state_dict + architecture

        # Fallback: rebuild the architecture from configs/model.yaml and load weights.
        # This imports the *architecture definition* only (no training loop, no
        # dataset manager, no optimizer) — consistent with "consume, don't retrain".
        from training.models import ModelFactory  # local import: optional dependency

        model = ModelFactory.from_config(model_config)
        state = torch.load(str(weights_path), map_location=self.device)
        state_dict = state.get("model_state_dict", state) if isinstance(state, dict) else state
        model.load_state_dict(state_dict)
        model.eval()
        model.to(self.device)
        return model, "state_dict"

    def _load_pickle(self, rel_path: str) -> Any:
        with self._path(rel_path).open("rb") as fh:
            return pickle.load(fh)

    def _load_json(self, rel_path: str) -> dict[str, Any]:
        return json.loads(self._path(rel_path).read_text(encoding="utf-8"))

    def _load_yaml(self, rel_path: str) -> dict[str, Any]:
        raw = yaml.safe_load(self._path(rel_path).read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _path(self, rel_path: str) -> Path:
        return self.package_dir / rel_path

    @staticmethod
    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device


__all__ = ["ReleasePackage", "ReleasePackageError", "ReleasePackageLoader"]
