"""Release validation (Phase R6).

:class:`ReleaseValidator` verifies a release package end to end:

* **integrity** — every required file exists and every file matches its
  SHA-256 in ``version/checksums.json``;
* **manifest** — the manifest schema is supported and consistent with the
  checksum file;
* **versions** — package / model / dataset / release versions are valid semver
  and agree with the release directory and the version files;
* **dependencies** — every Python dependency required to serve the release is
  importable (``onnxruntime`` only when the ONNX format is present);
* **compatibility** — model config <-> preprocessing pipeline agreement and
  fingerprint provenance;
* **smoke test** — the packaged PyTorch model restores and runs one forward
  pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from shared.exceptions import InvalidVersionError
from shared.utils.hash import sha256_file
from shared.versioning import SemanticVersion

from .config import RuntimeConfig, ValidationConfig
from .exceptions import (
    ReleaseValidationError,
    ReleaseNotFoundError,
)
from .layout import (
    FORMAT_FILES,
    REQUIRED_RELEASE_FILES,
    ReleaseLayout,
    ReleaseManifest,
    parse_release_dir,
    resolve_release,
)

#: (module, required) — dependencies the runtime always needs.
REQUIRED_DEPENDENCIES: tuple[tuple[str, str], ...] = (
    ("torch", "PyTorch (pytorch / torchscript backends)"),
    ("numpy", "numpy tensors"),
    ("pandas", "metadata dataframes"),
    ("pyarrow", "parquet reads"),
    ("yaml", "configuration parsing"),
)
#: Dependencies needed only when a specific model format is shipped.
FORMAT_DEPENDENCIES: dict[str, tuple[str, str]] = {
    "onnx": ("onnxruntime", "ONNX inference"),
}


@dataclass
class ReleaseValidationResult:
    """Result of a release validation run."""

    valid: bool
    checks: dict[str, bool] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    version: str | None = None
    release_path: Path | None = None
    backend: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "checks": self.checks,
            "errors": self.errors,
            "warnings": self.warnings,
            "version": self.version,
            "release_path": str(self.release_path) if self.release_path else None,
            "backend": self.backend,
        }


class ReleaseValidator:
    """Validate a release package directory.

    Args:
        config: Validated :class:`RuntimeConfig` (``None`` = defaults).
    """

    def __init__(self, config: RuntimeConfig | None = None) -> None:
        self.config = config or RuntimeConfig()
        self.validation: ValidationConfig = self.config.validation

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def validate_release(
        self, release_path: str | Path, *, strict: bool | None = None
    ) -> ReleaseValidationResult:
        """Run the full validation battery over ``release_path``.

        Raises:
            ReleaseValidationError: When a check fails and ``strict`` is true
                (defaults to ``validation.strict``).
        """
        layout = ReleaseLayout(release_path)
        result = ReleaseValidationResult(
            valid=True, version=_version_of(layout), release_path=layout.root
        )

        if self.validation.verify_checksums:
            ok, errors = self.verify_integrity(layout)
            result.checks["integrity"] = ok
            result.errors.extend(errors)

        if self.validation.verify_manifest:
            ok, errors = self.verify_manifest(layout)
            result.checks["manifest"] = ok
            result.errors.extend(errors)

        if self.validation.verify_versions:
            ok, errors = self.verify_versions(layout)
            result.checks["versions"] = ok
            result.errors.extend(errors)

        if self.validation.verify_dependencies:
            ok, errors, warnings = self.verify_dependencies(layout)
            result.checks["dependencies"] = ok
            result.errors.extend(errors)
            result.warnings.extend(warnings)

        if self.validation.verify_compatibility:
            ok, errors = self.verify_compatibility(layout)
            result.checks["compatibility"] = ok
            result.errors.extend(errors)

        if self.validation.smoke_test and layout.has_format("pytorch"):
            ok, errors = self.smoke_test(layout)
            result.checks["smoke_test"] = ok
            result.errors.extend(errors)

        result.backend = self._pick_backend(layout)
        result.valid = all(result.checks.values()) and not result.errors
        if not result.valid and (self.validation.strict if strict is None else strict):
            raise ReleaseValidationError(
                f"release {result.version or '?'} failed validation",
                detail=result.to_dict(),
            )
        return result

    def validate_version(
        self, releases_root: str | Path, version: str, *, strict: bool | None = None
    ) -> ReleaseValidationResult:
        """Resolve and validate the release directory for ``version``."""
        try:
            path = resolve_release(releases_root, version)
        except ReleaseNotFoundError:
            raise
        return self.validate_release(path, strict=strict)

    # ------------------------------------------------------------------ #
    # Checks
    # ------------------------------------------------------------------ #

    def verify_integrity(self, layout: ReleaseLayout) -> tuple[bool, list[str]]:
        """Recompute SHA-256 for every file listed in the checksum map."""
        errors: list[str] = []
        checksums_path = layout.artifact("version/checksums.json")
        if not checksums_path.exists():
            return False, ["version/checksums.json is missing"]

        checksums = layout.checksums()
        ok = True
        # ``version/manifest.json`` and ``version/checksums.json`` cannot
        # checksum themselves; they are checked for existence only.
        self_referential = {"version/manifest.json", "version/checksums.json"}
        for rel in REQUIRED_RELEASE_FILES:
            if rel in self_referential:
                if not layout.exists(rel):
                    errors.append(f"{rel}: missing")
                    ok = False
                continue
            if rel not in checksums:
                errors.append(f"{rel}: not listed in checksums.json")
                ok = False
            elif not layout.exists(rel):
                errors.append(f"{rel}: missing")
                ok = False

        for rel, expected in checksums.items():
            path = layout.artifact(rel)
            if not path.exists():
                errors.append(f"{rel}: missing")
                ok = False
                continue
            actual = sha256_file(path)
            if actual != str(expected):
                errors.append(
                    f"{rel}: checksum mismatch (expected {expected}, got {actual})"
                )
                ok = False
        return ok, errors

    def verify_manifest(self, layout: ReleaseLayout) -> tuple[bool, list[str]]:
        """Check the release manifest schema + checksum consistency."""
        errors: list[str] = []
        manifest_path = layout.artifact("version/manifest.json")
        if not manifest_path.exists():
            return False, ["version/manifest.json is missing"]
        try:
            manifest = ReleaseManifest.load(layout.root)
        except Exception as exc:  # ReleaseLayoutError
            return False, [f"manifest unreadable: {exc}"]

        ok = True
        if manifest.manifest_version != 2:
            errors.append(f"unsupported manifest_version {manifest.manifest_version}")
            ok = False
        checksum_file = manifest.checksum_file
        if checksum_file != "version/checksums.json":
            errors.append(f"unexpected checksum_file {checksum_file!r}")
            ok = False
        if not layout.exists("version/checksums.json"):
            errors.append("version/checksums.json is missing")
            ok = False
        else:
            checksum_keys = set(layout.checksums())
            manifest_keys = set(manifest.files)
            if manifest_keys != checksum_keys:
                errors.append(
                    "manifest files do not match checksums.json "
                    f"(manifest={sorted(manifest_keys)}, "
                    f"checksums={sorted(checksum_keys)})"
                )
                ok = False
            self_referential = {"version/manifest.json", "version/checksums.json"}
            for rel in REQUIRED_RELEASE_FILES:
                if rel in self_referential:
                    continue
                if rel not in manifest_keys:
                    errors.append(f"manifest is missing required file {rel}")
                    ok = False
        for key in ("package_version", "model_version", "dataset_version",
                    "release_version"):
            try:
                SemanticVersion.from_string(str(getattr(manifest, key)))
            except (InvalidVersionError, ValueError):
                errors.append(
                    f"manifest {key} is not valid semver: "
                    f"{getattr(manifest, key)!r}"
                )
                ok = False
        return ok, errors

    def verify_versions(self, layout: ReleaseLayout) -> tuple[bool, list[str]]:
        """Cross-check versions across the manifest, dir name and version files."""
        errors: list[str] = []
        if not layout.exists("version/manifest.json"):
            return False, ["version/manifest.json is missing"]

        manifest = layout.manifest()
        version = _version_of(layout)
        ok = True

        parsed_dir = parse_release_dir(layout.root.name)
        if parsed_dir is not None and version is not None and parsed_dir != version:
            errors.append(
                f"release directory name version {parsed_dir} does not match "
                f"manifest package_version {version}"
            )
            ok = False

        if version != manifest.package_version:
            errors.append(
                f"manifest package_version {manifest.package_version} does not "
                f"match release version {version}"
            )
            ok = False
        if manifest.package_version != manifest.release_version:
            errors.append(
                "package_version and release_version disagree: "
                f"{manifest.package_version} vs {manifest.release_version}"
            )
            ok = False

        release_version_path = layout.artifact("version/release_version.json")
        if release_version_path.exists():
            payload = _load_json(release_version_path)
            recorded = str(payload.get("version") or "")
            if recorded and recorded != version:
                errors.append(
                    f"release_version.json version {recorded} does not match "
                    f"{version}"
                )
                ok = False
        return ok, errors

    def verify_dependencies(
        self, layout: ReleaseLayout
    ) -> tuple[bool, list[str], list[str]]:
        """Check that required Python dependencies are importable."""
        errors: list[str] = []
        warnings: list[str] = []
        for module, purpose in REQUIRED_DEPENDENCIES:
            if not _importable(module):
                errors.append(
                    f"required dependency {module!r} is not importable "
                    f"({purpose})"
                )
        for fmt, (module, purpose) in FORMAT_DEPENDENCIES.items():
            rel = FORMAT_FILES.get(fmt)
            if rel and layout.exists(rel) and not _importable(module):
                errors.append(
                    f"release ships {fmt} but required dependency "
                    f"{module!r} is not importable ({purpose})"
                )
        return (not errors), errors, warnings

    def verify_compatibility(self, layout: ReleaseLayout) -> tuple[bool, list[str]]:
        """Model config <-> preprocessing pipeline <-> fingerprint agreement."""
        errors: list[str] = []
        config_path = layout.artifact("configs/model_config.yaml")
        scalers_path = layout.artifact("preprocess/feature_scalers.pkl")
        encoder_path = layout.artifact("preprocess/label_encoder.pkl")
        for path in (config_path, scalers_path, encoder_path):
            if not path.exists():
                return False, [f"{path.name} is missing"]

        try:
            from training.models import ModelConfig

            config = ModelConfig.model_validate(
                yaml.safe_load(config_path.read_text(encoding="utf-8"))
            )
        except Exception as exc:  # yaml / pydantic errors
            return False, [f"model_config.yaml is unreadable: {exc}"]

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

        # Fingerprint provenance: the release manifest should agree with the
        # original (R5) manifest when it was copied in.
        original = layout.artifact("version/original_manifest.json")
        if original.exists():
            original_payload = _load_json(original)
            original_fp = original_payload.get("model_fingerprint")
            manifest_fp = layout.manifest().model_fingerprint
            if original_fp and manifest_fp and original_fp != manifest_fp:
                errors.append("model_fingerprint changed vs original manifest")

        return (not errors), errors

    def smoke_test(self, layout: ReleaseLayout) -> tuple[bool, list[str]]:
        """Restore the PyTorch model and run one warm-up forward pass."""
        model_path = layout.artifact("model/cropfusion.pt")
        if not model_path.exists():
            return False, ["model/cropfusion.pt is missing (smoke test skipped)"]
        try:
            import torch

            from training.inference import load_pytorch_model

            model, _ = load_pytorch_model(model_path)
            batch = _build_warmup_batch(model.config)
            with torch.no_grad():
                model(batch)
            return True, []
        except Exception as exc:  # noqa: BLE001 - report any failure
            return False, [f"smoke test failed: {exc}"]

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _pick_backend(layout: ReleaseLayout) -> str | None:
        if layout.has_format("pytorch"):
            return "pytorch"
        if layout.has_format("torchscript"):
            return "torchscript"
        if layout.has_format("onnx"):
            return "onnx"
        return None


def _version_of(layout: ReleaseLayout) -> str | None:
    if not layout.exists("version/manifest.json"):
        return None
    try:
        return layout.manifest().package_version
    except Exception:  # ReleaseLayoutError
        return None


def _format_rel(fmt: str) -> str | None:
    return FORMAT_FILES.get(fmt)


def _importable(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


def _load_json(path: Path) -> dict[str, Any]:
    import json

    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}


def _load_pickle(path: Path) -> Any:
    import pickle

    with open(path, "rb") as fh:
        return pickle.load(fh)


def _build_warmup_batch(cfg: Any) -> dict[str, Any]:
    """A deterministic zero / one batch built purely from the model config."""
    import torch

    batch: dict[str, Any] = {
        "tabular": torch.zeros(2, cfg.tabular_feature_dim, dtype=torch.float32),
    }
    if cfg.uses_image:
        seq_len = getattr(cfg.temporal, "max_len", 2) or 2
        size = cfg.image_encoder.input_size or 32
        batch["ndvi"] = torch.zeros(2, seq_len, 1, size, size)
        batch["evi"] = torch.zeros(2, seq_len, 1, size, size)
        batch["temporal_mask"] = torch.ones(2, seq_len, dtype=torch.bool)
    return batch
