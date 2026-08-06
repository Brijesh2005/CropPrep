"""Artifact versioning for the inference package (Phase R5).

Provides semantic-version resolution and content fingerprints for the model /
dataset / training-run provenance stamped into the inference package:

* :func:`git_commit` — the current repository HEAD (best effort),
* :func:`model_fingerprint` — a stable SHA-256 over the model architecture
  config + parameter values,
* :func:`content_sha256` — a canonical SHA-256 of a JSON-serialisable payload,
* :func:`resolve_versions` — validated semver for the package / model /
  dataset / training run,
* :func:`build_version_files` — the ``dataset_version.json`` /
  ``model_version.json`` payloads.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

from shared.versioning import SemanticVersion

from .exceptions import VersioningError


def git_commit(repo_root: str | Path | None = None) -> str | None:
    """Return the current repository HEAD SHA-256 (or ``None`` when absent)."""
    root = str(Path(repo_root or Path.cwd()))
    try:
        result = subprocess.run(
            ["git", "-C", root, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def content_sha256(payload: Any) -> str:
    """Canonical SHA-256 of a JSON-serialisable payload."""
    try:
        encoded = json.dumps(
            payload, sort_keys=True, default=_json_default
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise VersioningError(
            f"cannot hash payload of type {type(payload).__name__}", detail=exc
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    """SHA-256 of a file's bytes."""
    from shared.utils.hash import sha256_file

    return sha256_file(path)


def model_fingerprint(model: Any) -> str:
    """Stable SHA-256 of the model architecture + parameter values.

    Args:
        model: A ``CropFusionModel`` (exposes ``config`` + ``state_dict``).

    Returns:
        A hex digest covering the config and the concatenated parameter bytes.
    """
    config = getattr(model, "config", None)
    if config is None:
        raise VersioningError(
            "model_fingerprint requires a model exposing a ModelConfig"
        )
    try:
        state = model.state_dict()
    except Exception as exc:  # pragma: no cover - defensive
        raise VersioningError(f"cannot read model state: {exc}") from exc
    digest = hashlib.sha256()
    digest.update(content_sha256(config.model_dump()).encode("utf-8"))
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        digest.update(state[name].detach().cpu().contiguous().view(-1).numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ResolvedVersions:
    """Validated semver strings for the artifacts in a package."""

    package_version: str
    model_version: str
    dataset_version: str
    training_version: str | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("package", self.package_version),
            ("model", self.model_version),
            ("dataset", self.dataset_version),
        ):
            try:
                SemanticVersion.from_string(value)
            except Exception as exc:
                raise VersioningError(
                    f"invalid {name} version {value!r}",
                    detail=exc,
                    suggested_resolution="Use MAJOR.MINOR.PATCH, e.g. '1.0.0'",
                ) from exc


def resolve_versions(
    *,
    package_version: str = "1.0.0",
    model_version: str | None = None,
    dataset_version: str | None = None,
    model_config_version: str = "1.0.0",
    dataset_manager_version: str | None = None,
    training_version: str | None = None,
) -> ResolvedVersions:
    """Resolve and validate the semver strings for a package build."""
    return ResolvedVersions(
        package_version=package_version,
        model_version=model_version or model_config_version,
        dataset_version=dataset_version or dataset_manager_version or "1.0.0",
        training_version=training_version,
    )


def bump_semver(version: str, part: str = "patch") -> str:
    """Bump a ``MAJOR.MINOR.PATCH`` version (major / minor / patch)."""
    try:
        return str(SemanticVersion.from_string(version).bump(part))
    except Exception as exc:
        raise VersioningError(
            f"cannot bump version {version!r}", detail=exc
        ) from exc


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, (set, tuple)):
        return list(value)
    if isinstance(value, torch.Tensor):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value)}")


def build_version_files(
    resolved: ResolvedVersions,
    *,
    model_fingerprint: str,
    dataset_fingerprint: str,
    training_fingerprint: str | None = None,
    git_commit_sha: str | None = None,
    package_name: str = "cropfusion",
    extra: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the ``dataset_version.json`` / ``model_version.json`` payloads."""
    now = datetime.now(timezone.utc).isoformat()
    dataset = {
        "kind": "dataset",
        "name": f"{package_name}-dataset",
        "version": resolved.dataset_version,
        "status": "released",
        "checksum": dataset_fingerprint,
        "created_at": now,
        "message": "Dataset snapshot persisted into the inference package.",
        **(extra or {}),
    }
    model = {
        "kind": "model",
        "name": f"{package_name}-model",
        "version": resolved.model_version,
        "status": "released",
        "checksum": model_fingerprint,
        "created_at": now,
        "message": "Trained CropFusion multimodal model export.",
        "architecture_version": None,
        "training_version": training_fingerprint,
        "training_run_version": training_version_fingerprint(resolved, training_fingerprint),
        "git_commit": git_commit_sha,
        **(extra or {}),
    }
    return dataset, model


def training_version_fingerprint(
    resolved: ResolvedVersions, training_fingerprint: str | None
) -> str | None:
    """Fingerprint of the training run (semver + content hash, if provided)."""
    if training_fingerprint is None:
        return None
    return content_sha256(
        {
            "training_version": resolved.training_version,
            "fingerprint": training_fingerprint,
        }
    )
