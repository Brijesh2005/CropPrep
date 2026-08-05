"""Environment Manager for the Kaggle Training Platform (R2.1).

Facade over the environment probes (runtime, GPU/CUDA, host system,
dependencies). Builds the capability report consumed by the bootstrap script,
the Training Validator and the reports module.

Usage::

    from training.kaggle.config import load_paths_config
    from training.kaggle.environment import EnvironmentManager

    manager = EnvironmentManager()
    report = manager.report()          # full capability report
    manager.gpu()                      # GPU/CUDA section only

Pure infrastructure — no training logic.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from .dependencies import DEFAULT_PROBES, detect_dependencies
from .gpu import detect_gpu
from .runtime import detect_runtime
from .system import detect_system


class EnvironmentManager:
    """Builds environment capability reports for the Training Platform.

    Args:
        repo_root: Repository root used for disk-space measurement
            (defaults to the auto-detected repo root).
    """

    def __init__(self, repo_root: str | None = None) -> None:
        from pathlib import Path

        self.repo_root = (
            Path(repo_root).resolve()
            if repo_root
            else Path(__file__).resolve().parents[3]
        )

    # ------------------------------------------------------------------ #
    # Individual sections
    # ------------------------------------------------------------------ #

    def runtime(self) -> dict[str, Any]:
        """Kaggle runtime detection."""
        return detect_runtime()

    def gpu(self) -> dict[str, Any]:
        """GPU / CUDA capability report."""
        return detect_gpu()

    def system(self) -> dict[str, Any]:
        """Host system report (CPU / RAM / disk / python)."""
        return detect_system(self.repo_root)

    def dependencies(
        self, requirements: list[str] | None = None
    ) -> dict[str, Any]:
        """Dependency import/version probe report."""
        return detect_dependencies(requirements)

    # ------------------------------------------------------------------ #
    # Combined capability report
    # ------------------------------------------------------------------ #

    def report(self, requirements: list[str] | None = None) -> dict[str, Any]:
        """Full capability report: runtime + system + GPU + dependencies.

        Args:
            requirements: Dependency names to probe; defaults to
                :data:`DEFAULT_PROBES`.
        """
        deps = self.dependencies(requirements)
        gpu = self.gpu()
        missing = sorted(
            name for name, info in deps.items() if not info["installed"]
        )
        return {
            "runtime": self.runtime(),
            "system": self.system(),
            "gpu": gpu,
            "dependencies": deps,
            "capable": {
                "gpu": bool(gpu["available"]),
                "cuda": bool(gpu["cuda_available"]),
                "torch_cuda": bool(
                    gpu["cuda_available"] and importlib.util.find_spec("torch")
                ),
                "all_required_installed": not missing,
                "missing_dependencies": missing,
            },
        }
