"""Training Validator for the Kaggle Training Platform (R2.1).

Validates the Kaggle environment + workspace and returns a
:class:`shared.validation.ValidationResult`. Checks:

* configuration   — every registry config file exists and parses as YAML,
* python          — version satisfies ``environment.min_python``,
* GPU / CUDA      — available when ``require_gpu`` (incl. gpu dependencies),
* dependencies    — required packages importable + versioned,
* folder structure— workspace directories exist and are writable,
* permissions     — write access to outputs / checkpoints / logs,
* disk space      — free space >= ``min_free_gb`` on the outputs volume,
* dataset providers — optional: given provider manifests are available.

Pure infrastructure — no training logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from shared.enums import Severity
from shared.validation import ValidationIssue, ValidationResult

from .config import PathsConfig, WorkspaceLayout, load_yaml_document
from .workspace import WorkspaceManager

#: Config sections mapped to the registry keys they validate.
_CONFIG_KEYS = (
    "dataset",
    "training",
    "kaggle",
    "model",
    "logging",
    "paths",
    "validation",
)


class TrainingValidator:
    """Runs the infrastructure readiness checks for a Kaggle workspace.

    Args:
        paths: Validated :class:`PathsConfig` (requirements + registry).
        layout: Resolved :class:`WorkspaceLayout`.
        environment: Environment capability report (see
            :meth:`training.kaggle.environment.EnvironmentManager.report`).
    """

    def __init__(
        self,
        paths: PathsConfig,
        layout: WorkspaceLayout,
        environment: Mapping[str, Any],
    ) -> None:
        self.paths = paths
        self.layout = layout
        self.environment = environment
        self.workspace = WorkspaceManager(layout)

    # ------------------------------------------------------------------ #
    # Entry point
    # ------------------------------------------------------------------ #

    def validate(
        self,
        provider_manifests: Mapping[str, Any] | None = None,
    ) -> ValidationResult:
        """Run all checks and return an aggregated :class:`ValidationResult`."""
        issues: list[ValidationIssue] = []
        issues += self._check_configuration()
        issues += self._check_python()
        issues += self._check_gpu()
        issues += self._check_dependencies()
        issues += self._check_folders()
        issues += self._check_permissions()
        issues += self._check_disk()
        if provider_manifests is not None:
            issues += self._check_providers(provider_manifests)
        passed = not any(
            issue.severity in (Severity.ERROR, Severity.CRITICAL) for issue in issues
        )
        return ValidationResult(
            passed=passed,
            issues=issues,
            target="training/kaggle",
        )

    # ------------------------------------------------------------------ #
    # Individual checks
    # ------------------------------------------------------------------ #

    def _check_configuration(self) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        registry = self.paths.config
        for key in _CONFIG_KEYS:
            rel = getattr(registry, key)
            path = self.layout.repo_root / rel
            if not path.exists():
                issues.append(
                    ValidationIssue(
                        code="CONFIG_MISSING",
                        severity=Severity.ERROR,
                        message=f"Config file missing: {rel}",
                        path=str(path),
                    )
                )
                continue
            try:
                raw = load_yaml_document(path)
            except (OSError, ValueError) as exc:
                issues.append(
                    ValidationIssue(
                        code="CONFIG_INVALID",
                        severity=Severity.ERROR,
                        message=f"Config file unreadable: {rel}",
                        path=str(path),
                        detail=str(exc),
                    )
                )
                continue
            if raw is None or not isinstance(raw, dict):
                issues.append(
                    ValidationIssue(
                        code="CONFIG_ROOT",
                        severity=Severity.ERROR,
                        message=f"Config root must be a mapping: {rel}",
                        path=str(path),
                    )
                )
        if not issues:
            issues.append(
                ValidationIssue(
                    code="CONFIG_OK",
                    severity=Severity.INFO,
                    message="All registry config files exist and parse",
                )
            )
        return issues

    def _check_python(self) -> list[ValidationIssue]:
        system = self.environment.get("system", {})
        current = system.get("python_version")
        minimum = self.paths.environment.min_python
        ok = _version_tuple(current) >= _version_tuple(minimum)
        if not ok:
            return [
                ValidationIssue(
                    code="PYTHON_TOO_OLD",
                    severity=Severity.ERROR,
                    message=(
                        f"Python {current} does not satisfy minimum "
                        f"{minimum}"
                    ),
                )
            ]
        return [
            ValidationIssue(
                code="PYTHON_OK",
                severity=Severity.INFO,
                message=f"Python {current} >= {minimum}",
            )
        ]

    def _check_gpu(self) -> list[ValidationIssue]:
        gpu = self.environment.get("gpu", {})
        available = bool(gpu.get("available"))
        cuda = bool(gpu.get("cuda_available"))
        require = self.paths.environment.require_gpu
        issues: list[ValidationIssue] = []
        if require and not available:
            issues.append(
                ValidationIssue(
                    code="GPU_UNAVAILABLE",
                    severity=Severity.ERROR,
                    message="GPU required by configuration but not detected",
                )
            )
        elif available:
            issues.append(
                ValidationIssue(
                    code="GPU_OK",
                    severity=Severity.INFO,
                    message=f"GPU available (devices={gpu.get('device_count')})",
                )
            )
        if require and available and not cuda:
            issues.append(
                ValidationIssue(
                    code="CUDA_UNAVAILABLE",
                    severity=Severity.WARNING,
                    message="GPU present but CUDA not available via torch",
                )
            )
        return issues

    def _check_dependencies(self) -> list[ValidationIssue]:
        requirements = self.paths.environment.required_dependencies
        gpu_deps = self.paths.environment.gpu_dependencies
        deps = self.environment.get("dependencies", {})
        issues: list[ValidationIssue] = []
        missing = [
            name for name in requirements if not deps.get(name, {}).get("installed")
        ]
        if missing:
            issues.append(
                ValidationIssue(
                    code="DEPENDENCY_MISSING",
                    severity=Severity.ERROR,
                    message=f"Required dependencies not installed: {missing}",
                    detail=missing,
                )
            )
        if self.paths.environment.require_gpu:
            gpu_missing = [
                name
                for name in gpu_deps
                if not deps.get(name, {}).get("installed")
            ]
            if gpu_missing:
                issues.append(
                    ValidationIssue(
                        code="GPU_DEPENDENCY_MISSING",
                        severity=Severity.ERROR,
                        message=f"GPU dependencies not installed: {gpu_missing}",
                        detail=gpu_missing,
                    )
                )
        if not missing:
            issues.append(
                ValidationIssue(
                    code="DEPENDENCY_OK",
                    severity=Severity.INFO,
                    message="Required dependencies installed",
                )
            )
        return issues

    def _check_folders(self) -> list[ValidationIssue]:
        missing = [str(p) for p in self._required_dirs() if not p.exists()]
        if missing:
            return [
                ValidationIssue(
                    code="WORKSPACE_MISSING",
                    severity=Severity.ERROR,
                    message=f"Workspace folders missing: {missing}",
                    detail=missing,
                )
            ]
        return [
            ValidationIssue(
                code="WORKSPACE_OK",
                severity=Severity.INFO,
                message="Workspace folder structure present",
            )
        ]

    def _required_dirs(self) -> list[Path]:
        layout = self.layout
        return [
            layout.root,
            layout.logs,
            layout.outputs,
            layout.checkpoints,
            layout.cache,
            layout.configs,
        ]

    def _check_permissions(self) -> list[ValidationIssue]:
        if self.workspace.ensure():
            return [
                ValidationIssue(
                    code="PERMISSION_OK",
                    severity=Severity.INFO,
                    message="Workspace directories are writable",
                )
            ]
        return [
            ValidationIssue(
                code="PERMISSION_DENIED",
                severity=Severity.ERROR,
                message="One or more workspace directories are not writable",
            )
        ]

    def _check_disk(self) -> list[ValidationIssue]:
        from .environment.system import detect_disk

        disk = detect_disk(self.layout.outputs)
        free_gb = disk.get("free_gb")
        minimum = self.paths.environment.min_free_gb
        if free_gb is None:
            return [
                ValidationIssue(
                    code="DISK_UNKNOWN",
                    severity=Severity.WARNING,
                    message="Could not measure free disk space",
                )
            ]
        if free_gb < minimum:
            return [
                ValidationIssue(
                    code="DISK_LOW",
                    severity=Severity.ERROR,
                    message=(
                        f"Only {free_gb:.2f} GB free on outputs volume; "
                        f"minimum {minimum:.1f} GB"
                    ),
                )
            ]
        return [
            ValidationIssue(
                code="DISK_OK",
                severity=Severity.INFO,
                message=f"{free_gb:.2f} GB free >= {minimum:.1f} GB minimum",
            )
        ]

    def _check_providers(
        self, manifests: Mapping[str, Any]
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for name, manifest in manifests.items():
            available = bool(manifest.get("available")) if isinstance(manifest, dict) else False
            if not available:
                issues.append(
                    ValidationIssue(
                        code="PROVIDER_UNAVAILABLE",
                        severity=Severity.WARNING,
                        message=f"Dataset provider not available: {name}",
                        path=str(name),
                    )
                )
        if not issues:
            issues.append(
                ValidationIssue(
                    code="PROVIDER_OK",
                    severity=Severity.INFO,
                    message="All dataset providers available",
                )
            )
        return issues

    def report(self, provider_manifests: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Run :meth:`validate` and return the dict form of the result."""
        return self.validate(provider_manifests).to_dict()


def _version_tuple(value: str | None) -> tuple[int, ...]:
    """Parse ``MAJOR.MINOR.PATCH`` into a comparable tuple (0.0.0 fallback)."""
    if not value:
        return (0, 0, 0)
    parts: list[int] = []
    for segment in value.split("."):
        digits = "".join(ch for ch in segment if ch.isdigit())
        if digits:
            parts.append(int(digits))
        if len(parts) == 3:
            break
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])  # type: ignore[return-value]
