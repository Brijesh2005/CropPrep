"""Checkpoint Manager for the Kaggle Training Platform (R2.1).

Owns the checkpoint *directory layout*, resume resolution and versioned
metadata. It deliberately does **not** save or load model weights — the actual
serialization is handled later by the training engine (``training.training``
and ``training.models`` checkpoint managers). This component only tracks:

* folder layout + a ``metadata.json`` registry,
* ``latest`` / ``best`` / ``resume`` resolution,
* per-run version numbers (incrementing integers + semver-style tags).

Pure infrastructure — no model code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import count
from pathlib import Path
from typing import Any

from shared.versioning import SemanticVersion

#: Monotonic insertion sequence so "newest first" ordering is unambiguous
#: even when several entries share the same wall-clock timestamp (fast,
#: sub-tick registration loops).
_SEQUENCE = count()


@dataclass(slots=True)
class CheckpointEntry:
    """Metadata record for one checkpoint (no weights)."""

    run_name: str
    stage: str = "checkpoint"
    version: str = "v1"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    seq: int = field(default_factory=lambda: next(_SEQUENCE))
    epoch: int | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    path: str | None = None
    resume: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "stage": self.stage,
            "version": self.version,
            "created_at": self.created_at,
            "seq": self.seq,
            "epoch": self.epoch,
            "metrics": self.metrics,
            "path": self.path,
            "resume": self.resume,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CheckpointEntry":
        return cls(
            run_name=data["run_name"],
            stage=data.get("stage", "checkpoint"),
            version=data.get("version", "v1"),
            created_at=data.get("created_at", ""),
            seq=data.get("seq") if data.get("seq") is not None else next(_SEQUENCE),
            epoch=data.get("epoch"),
            metrics=data.get("metrics", {}),
            path=data.get("path"),
            resume=data.get("resume", False),
        )


class CheckpointManager:
    """Tracks checkpoint metadata + layout for a Kaggle workspace.

    Args:
        checkpoint_dir: Checkpoints directory (from the workspace layout).
        keep_last: Maximum metadata entries retained per run (None = keep all).
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        keep_last: int | None = 20,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.keep_last = keep_last
        self.metadata_file = self.checkpoint_dir / "metadata.json"
        self._entries: list[CheckpointEntry] | None = None

    # ------------------------------------------------------------------ #
    # Layout / registry
    # ------------------------------------------------------------------ #

    def ensure_layout(self) -> Path:
        """Create the checkpoint directory + empty metadata registry."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if not self.metadata_file.exists():
            self._write([])
        return self.checkpoint_dir

    def _read(self) -> list[CheckpointEntry]:
        if self._entries is not None:
            return self._entries
        if self.metadata_file.exists():
            with self.metadata_file.open(encoding="utf-8") as fh:
                raw = json.load(fh)
            self._entries = [CheckpointEntry.from_dict(item) for item in raw]
        else:
            self._entries = []
        return self._entries

    def _write(self, entries: list[CheckpointEntry]) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        with self.metadata_file.open("w", encoding="utf-8") as fh:
            json.dump([e.to_dict() for e in entries], fh, indent=2, default=str)
        self._entries = entries

    # ------------------------------------------------------------------ #
    # Registry API
    # ------------------------------------------------------------------ #

    def list(self, run_name: str | None = None) -> list[dict[str, Any]]:
        """All tracked checkpoints, newest first (optionally per run)."""
        entries = self._read()
        if run_name is not None:
            entries = [e for e in entries if e.run_name == run_name]
        entries = sorted(entries, key=lambda e: (e.created_at, e.seq), reverse=True)
        return [e.to_dict() for e in entries]

    def latest(self, run_name: str | None = None) -> dict[str, Any] | None:
        """Newest registered checkpoint (optionally per run)."""
        entries = self.list(run_name)
        return entries[0] if entries else None

    def best(
        self,
        metric: str = "val_loss",
        mode: str = "min",
        run_name: str | None = None,
    ) -> dict[str, Any] | None:
        """Checkpoint with the best value for ``metric`` (min or max)."""
        candidates = [
            e for e in self._read() if metric in e.metrics
        ]
        if run_name is not None:
            candidates = [e for e in candidates if e.run_name == run_name]
        if not candidates:
            return None
        best_entry = max if mode == "max" else min
        chosen = best_entry(candidates, key=lambda e: e.metrics[metric])
        return chosen.to_dict()

    def resume(self, run_name: str | None = None) -> dict[str, Any] | None:
        """Checkpoint flagged for resume, else the latest one."""
        entries = self._read()
        if run_name is not None:
            entries = [e for e in entries if e.run_name == run_name]
        for entry in sorted(entries, key=lambda e: (e.created_at, e.seq), reverse=True):
            if entry.resume:
                return entry.to_dict()
        return self.latest(run_name)

    def register(
        self,
        run_name: str,
        *,
        stage: str = "checkpoint",
        epoch: int | None = None,
        metrics: dict[str, Any] | None = None,
        path: str | Path | None = None,
        resume: bool = False,
    ) -> dict[str, Any]:
        """Register a checkpoint metadata entry (no weights written)."""
        entry = CheckpointEntry(
            run_name=run_name,
            stage=stage,
            version=self._next_version(run_name, stage),
            epoch=epoch,
            metrics=metrics or {},
            path=str(path) if path is not None else None,
            resume=resume,
        )
        entries = self._read()
        entries.append(entry)
        if self.keep_last is not None and self.keep_last > 0:
            same_run = [e for e in entries if e.run_name == run_name]
            if len(same_run) > self.keep_last:
                keep = {id(e) for e in sorted(
                    same_run, key=lambda e: (e.created_at, e.seq), reverse=True
                )[: self.keep_last]}
                entries = [e for e in entries if e.run_name != run_name or id(e) in keep]
        self._write(entries)
        return entry.to_dict()

    def _next_version(self, run_name: str, stage: str) -> str:
        count = sum(1 for e in self._read() if e.run_name == run_name)
        return f"v{count + 1}"

    # ------------------------------------------------------------------ #
    # Versioning
    # ------------------------------------------------------------------ #

    def version_for(self, run_name: str) -> str:
        """Latest semver-style version tag for a run (``v1.0.0`` first)."""
        from shared.exceptions import InvalidVersionError

        versions = []
        for entry in self._read():
            if entry.run_name != run_name:
                continue
            try:
                versions.append(SemanticVersion.from_string(entry.version))
            except InvalidVersionError:
                continue
        if not versions:
            return "1.0.0"
        return str(max(versions).bump("patch"))

    def report(self) -> dict[str, Any]:
        """Checkpoint report: layout + summary + entries."""
        self.ensure_layout()
        entries = self.list()
        return {
            "checkpoint_dir": str(self.checkpoint_dir),
            "metadata_file": str(self.metadata_file),
            "count": len(entries),
            "runs": sorted({e["run_name"] for e in entries}),
            "latest": self.latest(),
            "resume": self.resume(),
            "entries": entries,
        }
