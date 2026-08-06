"""Inference package validation (Phase R5).

:class:`InferencePackageValidator` verifies a built (or received) inference
package end to end:

* **file integrity** — every required artifact exists and every file matches
  its SHA-256 in ``checksums.json``;
* **manifest consistency** — the manifest's file list matches the checksums
  and its version strings are valid semver;
* **model / config / dataset compatibility** — the model config's tabular
  feature dimension matches the shipped ``feature_scalers.pkl`` and its crop
  class count matches ``label_encoder.pkl``;
* **smoke test** — the packaged PyTorch model is restored and a forward pass
  runs without error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from shared.versioning import SemanticVersion

from .config import InferenceConfig
from .exceptions import PackageValidationError
from .versioning import file_sha256

#: Files that are validated for integrity but are not required (model formats).
OPTIONAL_MODEL_FORMATS = ("cropfusion.torchscript.pt", "cropfusion.onnx")


@dataclass
class ValidationResult:
    """Result of an inference-package validation."""

    valid: bool
    checks: dict[str, bool] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "checks": self.checks,
            "errors": self.errors,
        }


class InferencePackageValidator:
    """Validate an inference package directory.

    Args:
        config: Validated :class:`InferenceConfig` (``None`` = defaults).
    """

    def __init__(self, config: InferenceConfig | None = None) -> None:
        self.config = config or InferenceConfig()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def validate_package(self, directory: str | Path) -> ValidationResult:
        """Run the full validation battery over ``directory``.

        Raises:
            PackageValidationError: When a validation step fails outright
                (strict mode).
        """
        out_dir = Path(directory)
        result = ValidationResult(valid=True)

        integrity_ok, integrity_errors = self.verify_checksums(out_dir)
        result.checks["integrity"] = integrity_ok
        result.errors.extend(integrity_errors)

        manifest_ok, manifest_errors = self.validate_manifest(out_dir)
        result.checks["manifest"] = manifest_ok
        result.errors.extend(manifest_errors)

        compat_ok, compat_errors = self.validate_compatibility(out_dir)
        result.checks["compatibility"] = compat_ok
        result.errors.extend(compat_errors)

        if self.config.validation.smoke_test:
            smoke_ok, smoke_errors = self.smoke_test(out_dir)
            result.checks["smoke_test"] = smoke_ok
            result.errors.extend(smoke_errors)

        result.valid = all(result.checks.values()) and not result.errors
        if not result.valid and self.config.validation.strict:
            raise PackageValidationError(
                "inference package failed validation",
                detail={"errors": result.errors, "checks": result.checks},
            )
        return result

    # ------------------------------------------------------------------ #
    # Checks
    # ------------------------------------------------------------------ #

    def verify_checksums(self, directory: str | Path) -> tuple[bool, list[str]]:
        """Recompute SHA-256 for every file listed in ``checksums.json``."""
        out_dir = Path(directory)
        checksums_path = out_dir / "checksums.json"
        errors: list[str] = []
        if not checksums_path.exists():
            errors.append("checksums.json is missing")
            return False, errors

        from .package_builder import REQUIRED_ARTIFACTS

        checksums = _load_json(checksums_path)
        if not isinstance(checksums, dict):
            errors.append("checksums.json must be a mapping")
            return False, errors

        ok = True
        for name, expected in checksums.items():
            path = out_dir / name
            if not path.exists():
                errors.append(f"{name}: missing")
                ok = False
                continue
            actual = file_sha256(path)
            if actual != str(expected):
                errors.append(
                    f"{name}: checksum mismatch "
                    f"(expected {expected}, got {actual})"
                )
                ok = False

        for name in REQUIRED_ARTIFACTS:
            if name == "checksums.json":
                # ``checksums.json`` cannot checksum itself.
                if not (out_dir / name).exists():
                    errors.append(f"{name}: missing")
                    ok = False
                continue
            if name not in checksums:
                errors.append(f"{name}: not listed in checksums.json")
                ok = False
            elif not (out_dir / name).exists():
                errors.append(f"{name}: missing")
                ok = False
        return ok, errors

    def validate_manifest(self, directory: str | Path) -> tuple[bool, list[str]]:
        """Check manifest consistency (file list + valid semver strings)."""
        out_dir = Path(directory)
        errors: list[str] = []
        manifest_path = out_dir / "manifest.json"
        checksums_path = out_dir / "checksums.json"
        if not manifest_path.exists():
            return False, ["manifest.json is missing"]

        manifest = _load_json(manifest_path)
        checksums = _load_json(checksums_path) if checksums_path.exists() else {}

        ok = True
        if manifest.get("manifest_version") != 1:
            errors.append("unsupported manifest_version")
            ok = False
        files = manifest.get("files")
        if not isinstance(files, dict):
            errors.append("manifest has no files mapping")
            ok = False
        elif set(files.keys()) != set(checksums.keys()):
            errors.append(
                "manifest files do not match checksums.json "
                f"(manifest={sorted(files)}, checksums={sorted(checksums)})"
            )
            ok = False

        for key in ("package_version", "model_version", "dataset_version"):
            value = manifest.get(key)
            try:
                SemanticVersion.from_string(str(value))
            except Exception:
                errors.append(f"manifest {key} is not valid semver: {value!r}")
                ok = False
        return ok, errors

    def validate_compatibility(self, directory: str | Path) -> tuple[bool, list[str]]:
        """Model <-> preprocessor <-> config compatibility."""
        out_dir = Path(directory)
        errors: list[str] = []

        config_path = out_dir / "model_config.yaml"
        scalers_path = out_dir / "feature_scalers.pkl"
        encoder_path = out_dir / "label_encoder.pkl"
        for path in (config_path, scalers_path, encoder_path):
            if not path.exists():
                return False, [f"{path.name} is missing"]

        from training.models import ModelConfig

        config = ModelConfig.model_validate(
            _load_yaml(config_path)
        )

        scalers = _load_pickle(scalers_path)
        feature_names = getattr(scalers, "feature_names", None)
        if feature_names is not None and config.tabular_feature_dim != len(feature_names):
            errors.append(
                f"tabular feature mismatch: config={config.tabular_feature_dim} "
                f"vs scaler={len(feature_names)}"
            )

        encoder = _load_pickle(encoder_path)
        num_classes = int(getattr(encoder, "num_classes", 0) or 0)
        crop_classes = (
            config.heads.crop.num_classes
            if config.heads.crop is not None else 0
        )
        if num_classes and crop_classes and num_classes != crop_classes:
            errors.append(
                f"crop class mismatch: config={crop_classes} vs "
                f"label encoder={num_classes}"
            )
        return (not errors), errors

    def smoke_test(self, directory: str | Path) -> tuple[bool, list[str]]:
        """Restore the PyTorch model and run one forward pass."""
        out_dir = Path(directory)
        model_path = out_dir / "cropfusion.pt"
        if not model_path.exists():
            return False, ["cropfusion.pt is missing (smoke test skipped)"]
        try:
            import torch

            from .exporter import load_pytorch_model

            model, _ = load_pytorch_model(model_path)
            cfg = model.config
            batch = {
                "tabular": torch.zeros(
                    2, cfg.tabular_feature_dim, dtype=torch.float32
                ),
                "crop_label": torch.zeros(2, dtype=torch.long),
                "yield_label": torch.zeros(2, 1, dtype=torch.float32),
            }
            if cfg.uses_image:
                seq_len = getattr(cfg.temporal, "max_len", 2) or 2
                size = cfg.image_encoder.input_size or 32
                batch["ndvi"] = torch.zeros(2, seq_len, 1, size, size)
                batch["evi"] = torch.zeros(2, seq_len, 1, size, size)
                batch["temporal_mask"] = torch.ones(2, seq_len, dtype=torch.bool)
            with torch.no_grad():
                model(batch)
            return True, []
        except Exception as exc:  # noqa: BLE001 - report any failure
            return False, [f"smoke test failed: {exc}"]


def _load_json(path: Path) -> Any:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> Any:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_pickle(path: Path) -> Any:
    import pickle

    with open(path, "rb") as fh:
        return pickle.load(fh)


__all__ = ["InferencePackageValidator", "ValidationResult"]
