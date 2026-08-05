"""Filesystem model registry with a draft -> staging -> production lifecycle.

A registry entry is a directory ``<registry_dir>/<name>/<version>/`` containing
``manifest.json`` plus the model artifact(s). The manifest holds metrics,
hyperparameters, provenance and gate results. Promotion is gated by
:mod:`training.mlops.gates`.
"""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel

from .config import MLOpsSettings
from .gates import GateResult


class RegistryError(RuntimeError):
    pass


class ModelManifest(BaseModel):
    """Metadata for one registered model version."""

    model_config = {"extra": "forbid"}

    id: str
    name: str
    version: str
    status: str = "draft"  # draft | staging | production | archived
    checkpoint_path: str = ""
    metrics: dict[str, float] = {}
    hyperparameters: dict[str, Any] = {}
    git_commit: str | None = None
    created_at: str
    promoted_at: str | None = None
    promoted_by: str | None = None
    notes: str | None = None
    gates: list[dict[str, Any]] = field(default_factory=list)  # type: ignore[assignment]


@dataclass
class ModelRecord:
    """Convenience wrapper: manifest + storage locations."""

    manifest: ModelManifest
    dir: Path

    @property
    def name(self) -> str:
        return self.manifest.name

    @property
    def version(self) -> str:
        return self.manifest.version

    @property
    def status(self) -> str:
        return self.manifest.status

    def manifest_path(self) -> Path:
        return self.dir / "manifest.json"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class ModelRegistry:
    """Manage model versions on the filesystem."""

    VALID_STATUSES = {"draft", "staging", "production", "archived"}

    def __init__(self, settings: MLOpsSettings) -> None:
        self.settings = settings
        self.root = Path(settings.registry_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Registration
    # ------------------------------------------------------------------ #

    def register(
        self,
        name: str,
        version: str,
        *,
        checkpoint_path: str | Path,
        metrics: dict[str, float] | None = None,
        hyperparameters: dict[str, Any] | None = None,
        git_commit: str | None = None,
        notes: str | None = None,
    ) -> ModelRecord:
        """Create a new ``draft`` version. Raises if the version exists."""
        dest = self.root / name / version
        if (dest / "manifest.json").exists():
            raise RegistryError(f"version {name}@{version} already registered")
        dest.mkdir(parents=True, exist_ok=True)

        src = Path(checkpoint_path)
        if src.exists() and src.is_file():
            artifact = dest / src.name
            shutil.copy2(src, artifact)
            stored_path = str(artifact)
        else:
            stored_path = str(src)

        manifest = ModelManifest(
            id=uuid.uuid4().hex[:12],
            name=name,
            version=version,
            checkpoint_path=stored_path,
            metrics=metrics or {},
            hyperparameters=hyperparameters or {},
            git_commit=git_commit,
            created_at=_utcnow(),
            notes=notes,
        )
        self._write(manifest, dest)
        return ModelRecord(manifest, dest)

    # ------------------------------------------------------------------ #
    # Lookup
    # ------------------------------------------------------------------ #

    def list(
        self, name: str | None = None, status: str | None = None
    ) -> list[ModelRecord]:
        if status and status not in self.VALID_STATUSES:
            raise RegistryError(f"invalid status: {status}")
        records: list[ModelRecord] = []
        for manifest_path in sorted(self.root.rglob("manifest.json")):
            manifest = ModelManifest.model_validate(
                json.loads(manifest_path.read_text(encoding="utf-8"))
            )
            if name and manifest.name != name:
                continue
            if status and manifest.status != status:
                continue
            records.append(ModelRecord(manifest, manifest_path.parent))
        return records

    def get(self, name: str, version: str) -> ModelRecord:
        path = self.root / name / version / "manifest.json"
        if not path.exists():
            raise RegistryError(f"no such version: {name}@{version}")
        manifest = ModelManifest.model_validate(json.loads(path.read_text(encoding="utf-8")))
        return ModelRecord(manifest, path.parent)

    def active(self, name: str) -> ModelRecord | None:
        """The current production version for a model, if any."""
        prod = self.list(name=name, status="production")
        if not prod:
            return None
        return sorted(prod, key=lambda r: r.manifest.promoted_at or "", reverse=True)[0]

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def promote(
        self,
        name: str,
        version: str,
        *,
        target: str = "production",
        gates: Iterable[GateResult] | None = None,
        promoted_by: str | None = None,
    ) -> ModelRecord:
        """Run gates (optional) then move the version to ``target``.

        Promoting to ``production`` demotes the current incumbent to ``staging``
        and archives old production versions beyond the keep count.
        """
        if target not in {"staging", "production"}:
            raise RegistryError(f"invalid promotion target: {target}")
        record = self.get(name, version)
        if record.manifest.status == "archived":
            raise RegistryError("archived versions cannot be promoted")

        gate_results: list[GateResult] = list(gates or [])

        if target == "production":
            incumbent = self.active(name)
            if incumbent is not None and incumbent.version == version:
                raise RegistryError(f"{name}@{version} is already the active production model")

        manifest = record.manifest
        manifest.status = target
        manifest.promoted_at = _utcnow()
        manifest.promoted_by = promoted_by
        manifest.gates = [g.result() for g in gate_results]
        self._write(manifest, record.dir)

        if target == "production":
            self._prune_incumbent(name, version)
            self._archive_old(name)
        return self.get(name, version)

    def rollback(
        self, name: str, version: str, *, promoted_by: str | None = None
    ) -> ModelRecord:
        """Roll the active production version back to ``version``."""
        target = self.get(name, version)
        if target.manifest.status == "archived":
            raise RegistryError("cannot roll back to an archived version")
        active = self.active(name)
        if active is None:
            raise RegistryError(f"no active production model for {name} to roll back from")
        if active.version == version:
            raise RegistryError("requested rollback version is already active")

        active_manifest = active.manifest
        active_manifest.status = "staging"
        self._write(active_manifest, active.dir)

        target_manifest = target.manifest
        target_manifest.status = "production"
        target_manifest.promoted_at = _utcnow()
        target_manifest.promoted_by = promoted_by
        self._write(target_manifest, target.dir)
        return self.get(name, version)

    def archive(self, name: str, version: str, *, notes: str | None = None) -> ModelRecord:
        record = self.get(name, version)
        if record.manifest.status == "production":
            raise RegistryError("demote the production model before archiving")
        record.manifest.status = "archived"
        record.manifest.notes = notes or record.manifest.notes
        self._write(record.manifest, record.dir)
        return record

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _write(self, manifest: ModelManifest, dest: Path) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "manifest.json").write_text(
            manifest.model_dump_json(indent=2), encoding="utf-8"
        )

    def _prune_incumbent(self, name: str, promoted_version: str) -> None:
        for record in self.list(name=name, status="production"):
            if record.version != promoted_version:
                record.manifest.status = "staging"
                self._write(record.manifest, record.dir)

    def _archive_old(self, name: str) -> None:
        staging = sorted(
            self.list(name=name, status="staging"),
            key=lambda r: r.manifest.promoted_at or "",
            reverse=True,
        )
        keep = self.settings.keep_production_versions - 1
        for record in staging[keep:]:
            record.manifest.status = "archived"
            self._write(record.manifest, record.dir)
