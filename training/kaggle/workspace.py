"""Workspace Manager for the Kaggle Training Platform (R2.1).

Owns the Kaggle workspace folder structure (root, logs, outputs, checkpoints,
cache, configs), cache cleaning, resume resolution and checkpoint delegation.

Pure infrastructure — no training logic.
"""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from .cache import TrainingCache
from .checkpoints import CheckpointManager
from .config import WorkspaceLayout


class WorkspaceManager:
    """Creates and manages a Kaggle training workspace.

    Args:
        layout: Resolved :class:`WorkspaceLayout`.
        keep_last: Checkpoint metadata entries retained per run.
    """

    def __init__(
        self,
        layout: WorkspaceLayout,
        *,
        keep_last: int | None = 20,
    ) -> None:
        self.layout = layout
        self.checkpoints = CheckpointManager(layout.checkpoints, keep_last=keep_last)
        self.cache = TrainingCache(layout.cache)

    # ------------------------------------------------------------------ #
    # Folder structure
    # ------------------------------------------------------------------ #

    def create(self) -> dict[str, str]:
        """Create every workspace directory; returns the layout mapping."""
        for path in (
            self.layout.root,
            self.layout.logs,
            self.layout.outputs,
            self.layout.checkpoints,
            self.layout.cache,
            self.layout.configs,
        ):
            path.mkdir(parents=True, exist_ok=True)
        self.checkpoints.ensure_layout()
        self.cache.ensure_layout()
        return self.layout.as_dict()

    def ensure(self) -> bool:
        """True when every workspace directory exists and is writable."""
        try:
            self.create()
        except OSError:
            return False
        return all(_is_writable(p) for p in self._directories())

    def _directories(self) -> list[Path]:
        return [
            self.layout.root,
            self.layout.logs,
            self.layout.outputs,
            self.layout.checkpoints,
            self.layout.cache,
            self.layout.configs,
        ]

    # ------------------------------------------------------------------ #
    # Cache cleaning
    # ------------------------------------------------------------------ #

    def clean_cache(self, older_than_days: int | None = None) -> int:
        """Remove cached files; returns the number removed.

        Args:
            older_than_days: Only remove files older than this many days
                (None = remove every cache entry + files).
        """
        removed = self.cache.clear()
        cutoff = None
        if older_than_days is not None:
            cutoff = datetime.now().timestamp() - older_than_days * 86400
        for path in self.layout.cache.rglob("*"):
            if path.is_file():
                if cutoff is None or path.stat().st_mtime < cutoff:
                    try:
                        path.unlink()
                        removed += 1
                    except OSError:
                        continue
        return removed

    # ------------------------------------------------------------------ #
    # Outputs / temp
    # ------------------------------------------------------------------ #

    def output_path(self, *parts: str) -> Path:
        """Absolute path under the workspace ``outputs`` directory."""
        return self.layout.outputs.joinpath(*parts)

    def run_output(self, run_name: str) -> Path:
        """Per-run output directory (created on demand)."""
        path = self.output_path(run_name)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def temp_dir(self) -> Path:
        """A fresh temporary directory inside the workspace temp area."""
        return Path(tempfile.mkdtemp(dir=self.layout.outputs / "tmp"))

    def configs_dir(self) -> Path:
        """Resolved-config snapshot directory (created on demand)."""
        path = self.layout.configs
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ------------------------------------------------------------------ #
    # Checkpoints / resume
    # ------------------------------------------------------------------ #

    def resolve_resume(self, run_name: str | None = None) -> dict[str, Any] | None:
        """Delegates to :meth:`CheckpointManager.resume`."""
        return self.checkpoints.resume(run_name)

    def report(self) -> dict[str, Any]:
        """Workspace report: layout + cache + checkpoint summary."""
        self.create()
        return {
            "layout": self.layout.as_dict(),
            "cache": self.cache.stats(),
            "checkpoints": self.checkpoints.report(),
        }


def _is_writable(path: Path) -> bool:
    import os

    try:
        probe = path / f".write-test-{os.getpid()}"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False
