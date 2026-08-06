"""Release package layout (Phase R6).

A **release package** is the immutable, deployable unit produced from the
Phase R5 inference package. Every release directory follows one canonical
layout::

    cropfusion_release-<version>/
    ├── README.md
    ├── model/                      exported model artefacts
    │   ├── cropfusion.pt
    │   ├── cropfusion.torchscript.pt     (optional)
    │   ├── cropfusion.onnx               (optional)
    │   └── model_metadata.json
    ├── preprocess/                 fitted pipelines
    │   ├── feature_scalers.pkl
    │   ├── label_encoder.pkl
    │   └── preprocess_metadata.json
    ├── metadata/                   predict-only data artefacts
    │   ├── metadata.db
    │   ├── historical_context.parquet
    │   ├── location_index.parquet
    │   └── feature_lookup.parquet
    ├── configs/                    resolved configuration
    │   ├── model_config.yaml
    │   └── training_config.yaml
    ├── reports/                    evaluation / validation artefacts
    │   ├── metrics.json
    │   └── validation_report.json
    └── version/                    versioning + integrity
        ├── manifest.json
        ├── checksums.json
        ├── release_version.json
        ├── model_version.json
        ├── dataset_version.json
        ├── original_checksums.json      (R5 provenance)
        └── original_manifest.json       (R5 provenance)

This module defines the layout constants, the :class:`ReleaseLayout`
resolver, the :class:`ReleaseManifest` schema and :class:`ReleaseInfo`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml
from pydantic import BaseModel, ConfigDict

from shared.exceptions import InvalidVersionError
from shared.versioning import SemanticVersion

from .exceptions import ReleaseLayoutError, ReleaseNotFoundError

#: Name prefix of every release directory under the releases root.
RELEASE_DIR_PREFIX = "cropfusion_release"

#: Sub-directories of a release package (the canonical layout).
RELEASE_DIRS: tuple[str, ...] = (
    "model",
    "preprocess",
    "metadata",
    "configs",
    "reports",
    "version",
)

#: Files required in every release package (relative to the release root).
REQUIRED_RELEASE_FILES: tuple[str, ...] = (
    "README.md",
    "model/cropfusion.pt",
    "preprocess/feature_scalers.pkl",
    "preprocess/label_encoder.pkl",
    "metadata/metadata.db",
    "metadata/historical_context.parquet",
    "metadata/location_index.parquet",
    "metadata/feature_lookup.parquet",
    "configs/model_config.yaml",
    "configs/training_config.yaml",
    "reports/metrics.json",
    "version/manifest.json",
    "version/checksums.json",
)

#: Optional model formats (only present when exported).
OPTIONAL_MODEL_FORMATS: tuple[str, ...] = (
    "model/cropfusion.torchscript.pt",
    "model/cropfusion.onnx",
)

#: Model files by format name (mirrors the R5 bundle formats).
FORMAT_FILES: dict[str, str] = {
    "pytorch": "model/cropfusion.pt",
    "torchscript": "model/cropfusion.torchscript.pt",
    "onnx": "model/cropfusion.onnx",
}

#: Files that are copied as R5 provenance (flat-package checksums).
PROVENANCE_FILES: tuple[str, ...] = (
    "version/original_checksums.json",
    "version/original_manifest.json",
)


class ReleaseManifest(BaseModel):
    """Schema of ``version/manifest.json`` (release manifest version 2)."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    manifest_version: int = 2
    package_name: str
    package_version: str
    model_version: str
    dataset_version: str
    release_version: str
    created_at: str
    git_commit: str | None = None
    model_fingerprint: str = ""
    dataset_fingerprint: str = ""
    training_fingerprint: str | None = None
    formats: list[str] = field(default_factory=lambda: ["pytorch"])
    checksum_file: str = "version/checksums.json"
    required_files: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, root: str | Path) -> "ReleaseManifest":
        path = Path(root) / "version" / "manifest.json"
        if not path.exists():
            raise ReleaseLayoutError("release manifest is missing", detail=str(path))
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ReleaseLayoutError(
                "malformed release manifest YAML/JSON", detail=str(path)
            ) from exc
        if not isinstance(raw, dict):
            raise ReleaseLayoutError(
                "release manifest must be a mapping", detail=str(path)
            )
        return cls.model_validate(raw)


def parse_release_dir(name: str) -> str | None:
    """Extract the semver version from a release directory name.

    Accepts ``cropfusion_release-v1.2.3``, ``cropfusion_release_v1.2.3`` or a
    bare ``cropfusion_release`` (version then comes from the manifest).
    """
    if not name.startswith(RELEASE_DIR_PREFIX):
        return None
    rest = name[len(RELEASE_DIR_PREFIX):]
    if not rest:
        return None
    candidate = rest[1:] if rest[:1] in ("-", "_") else rest
    if candidate[:1] in ("v", "V"):
        candidate = candidate[1:]
    try:
        SemanticVersion.from_string(candidate)
    except InvalidVersionError:
        return None
    return candidate


def release_dir_name(version: str) -> str:
    """Canonical release directory name for a version."""
    try:
        SemanticVersion.from_string(version)
    except InvalidVersionError as exc:
        raise ReleaseLayoutError(
            f"invalid release version {version!r}", detail=version
        ) from exc
    return f"{RELEASE_DIR_PREFIX}-v{version}"


@dataclass
class ReleaseLayout:
    """Path resolver for a single release package.

    Args:
        root: The release package directory (e.g. ``releases/cropfusion_release-v1.0.0``).
    """

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    # -- Sub-directories --------------------------------------------------- #

    @property
    def model_dir(self) -> Path:
        return self.root / "model"

    @property
    def preprocess_dir(self) -> Path:
        return self.root / "preprocess"

    @property
    def metadata_dir(self) -> Path:
        return self.root / "metadata"

    @property
    def configs_dir(self) -> Path:
        return self.root / "configs"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def version_dir(self) -> Path:
        return self.root / "version"

    # -- Artifact resolution ------------------------------------------------ #

    def artifact(self, name: str) -> Path:
        """Resolve a relative artifact name (e.g. ``model/cropfusion.pt``)."""
        return self.root / name

    def exists(self, name: str) -> bool:
        return self.artifact(name).exists()

    def has_format(self, fmt: str) -> bool:
        rel = FORMAT_FILES.get(fmt)
        return rel is not None and self.exists(rel)

    @property
    def formats(self) -> list[str]:
        return [fmt for fmt in FORMAT_FILES if self.has_format(fmt)]

    def manifest(self) -> ReleaseManifest:
        return ReleaseManifest.load(self.root)

    def checksums(self) -> dict[str, str]:
        """Load the authoritative checksum map (relative path -> SHA-256)."""
        path = self.root / "version" / "checksums.json"
        if not path.exists():
            return {}
        import json

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ReleaseLayoutError(
                "malformed release checksums.json", detail=str(path)
            ) from exc
        if not isinstance(raw, dict):
            raise ReleaseLayoutError(
                "release checksums.json must be a mapping", detail=str(path)
            )
        return {str(k): str(v) for k, v in raw.items()}

    def is_valid_structure(self) -> tuple[bool, list[str]]:
        """Verify the directory layout + required files exist."""
        errors: list[str] = []
        for name in REQUIRED_RELEASE_FILES:
            if not self.exists(name):
                errors.append(f"{name}: missing")
        return (not errors), errors


@dataclass
class ReleaseInfo:
    """A discovered release package.

    Attributes:
        name: Release directory name.
        version: Semantic version string.
        path: Release package directory.
        manifest: Parsed :class:`ReleaseManifest` (``None`` when unreadable).
        model_version / dataset_version: Versions of the contained artefacts.
        formats: Export formats available in the release.
        discovered_at: UTC timestamp of discovery.
        structure_errors: Layout problems found at discovery time.
    """

    name: str
    version: str
    path: Path
    manifest: ReleaseManifest | None = None
    model_version: str | None = None
    dataset_version: str | None = None
    formats: list[str] = field(default_factory=list)
    discovered_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    structure_errors: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.structure_errors

    def to_dict(self) -> dict[str, Any]:
        manifest = self.manifest.model_dump() if self.manifest is not None else None
        return {
            "name": self.name,
            "version": self.version,
            "path": str(self.path),
            "manifest": manifest,
            "model_version": self.model_version,
            "dataset_version": self.dataset_version,
            "formats": self.formats,
            "discovered_at": self.discovered_at.isoformat(),
            "valid": self.valid,
            "structure_errors": self.structure_errors,
        }


def iter_release_dirs(root: str | Path) -> list[tuple[str, Path, str | None]]:
    """Return ``(name, path, version_or_None)`` for every release directory.

    A directory qualifies when its name starts with
    :data:`RELEASE_DIR_PREFIX` **and** it is a directory. The version is
    parsed from the name; a bare ``cropfusion_release`` yields ``None`` (the
    caller then reads the manifest).
    """
    base = Path(root)
    if not base.exists():
        return []
    found: list[tuple[str, Path, str | None]] = []
    for child in sorted(base.iterdir()):
        if not child.is_dir() or not child.name.startswith(RELEASE_DIR_PREFIX):
            continue
        version = parse_release_dir(child.name)
        found.append((child.name, child, version))
    return found


def resolve_release(root: str | Path, version: str) -> Path:
    """Resolve the release directory for ``version``.

    Raises:
        ReleaseNotFoundError: When no matching release directory exists.
    """
    for name, path, parsed in iter_release_dirs(root):
        if parsed == version:
            return path
    # Fall back to scanning manifests for the version.
    for name, path, parsed in iter_release_dirs(root):
        if parsed is not None:
            continue
        try:
            manifest = ReleaseManifest.load(path)
        except ReleaseLayoutError:
            continue
        if manifest.package_version == version:
            return path
    raise ReleaseNotFoundError(
        f"release version {version!r} not found under {root}",
        detail={"version": version, "root": str(root)},
    )
