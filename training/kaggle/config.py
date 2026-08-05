"""Configuration layer for the Kaggle Training Platform (R2.1).

Loads and validates the platform configuration with the same precedence as the
rest of CropFusion (see ``shared.config``):

1. Environment variables (``KAGGLE_<SECTION>__<KEY>``, nested via ``__``).
2. YAML configuration files (with ``extends`` inheritance for ``paths.yaml``).
3. Pydantic defaults.

The module owns three config documents:

* :class:`PathsConfig` — ``training/config/paths.yaml`` (workspace layout,
  config registry, environment requirements). Supports YAML ``extends`` chains.
* :class:`KaggleConfig` — ``training/config/kaggle.yaml`` (Kaggle runtime
  paths, dataset handle, editable install list).
* :class:`LoggingConfig` — ``training/config/logging.yaml`` (Training Logger).

This is pure configuration + path resolution — no training logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml
from pydantic import BaseModel, Field

from shared.config import deep_merge, parse_env

#: Environment prefix for the Kaggle Training Platform.
ENV_PREFIX = "KAGGLE_"

#: Default paths config location (repository-relative).
DEFAULT_PATHS_CONFIG = "training/config/paths.yaml"


class WorkspaceConfig(BaseModel):
    """Kaggle workspace layout (paths relative to the repository root)."""

    root: str = "training/kaggle"
    logs_dir: str = "logs"
    outputs_dir: str = "outputs"
    checkpoints_dir: str = "checkpoints"
    cache_dir: str = "cache"
    configs_dir: str = "configs"

    def snapshot(self) -> dict[str, Any]:
        return self.model_dump()


class ConfigRegistry(BaseModel):
    """Every Training Platform config file."""

    dataset: str = "training/config/dataset.yaml"
    training: str = "training/config/training.yaml"
    kaggle: str = "training/config/kaggle.yaml"
    model: str = "training/config/model.yaml"
    logging: str = "training/config/logging.yaml"
    paths: str = "training/config/paths.yaml"
    validation: str = "training/config/validation.yaml"

    def snapshot(self) -> dict[str, Any]:
        return self.model_dump()


class EnvironmentRequirements(BaseModel):
    """Minimum environment enforced by the Training Validator."""

    min_python: str = "3.10"
    require_gpu: bool = True
    min_free_gb: float = 5.0
    required_dependencies: list[str] = Field(
        default_factory=lambda: ["numpy", "pandas", "torch", "scikit-learn", "rasterio"]
    )
    gpu_dependencies: list[str] = Field(default_factory=lambda: ["torch"])


class PathsConfig(BaseModel):
    """Workspace layout + config registry for the Kaggle platform."""

    extends: str | None = None
    workspace: WorkspaceConfig = Field(default_factory=WorkspaceConfig)
    config: ConfigRegistry = Field(default_factory=ConfigRegistry)
    environment: EnvironmentRequirements = Field(default_factory=EnvironmentRequirements)


class KaggleRuntimeSection(BaseModel):
    """The ``kaggle:`` section of ``kaggle.yaml``."""

    dataset_handle: str = "shathanandabhatn/crop-yield-forecasting-karnataka-dakshina-kannada"
    competition: str | None = None
    gpu: bool = True
    internet: bool = True


class KaggleRuntimePaths(BaseModel):
    """The ``runtime:`` section of ``kaggle.yaml``."""

    repo_root: str | None = None
    input_dir: str = "/kaggle/input"
    working_dir: str = "/kaggle/working"
    artifacts_root: str = "/kaggle/working/artifacts"
    config_dir: str = "training/config"


class KaggleDatasetManagerSection(BaseModel):
    """The ``dataset_manager:`` section of ``kaggle.yaml``."""

    config_file: str = "training/config/dataset.yaml"
    dataset_root: str = "training/datasets"
    tabular_root: str = "training/datasets/tabular"


class KaggleOutputs(BaseModel):
    """The ``outputs:`` section of ``kaggle.yaml``."""

    run_dir: str = "training/artifacts/runs"
    checkpoint_dir: str = "training/artifacts/checkpoints"
    export_dir: str = "training/artifacts/export"
    release_dir: str = "training/artifacts/releases"


class KaggleInstall(BaseModel):
    """The ``install:`` section of ``kaggle.yaml`` (editable packages)."""

    editable_packages: list[str] = Field(
        default_factory=lambda: [
            "training/models",
            "training/preprocessing",
            "training/training",
            "training/dataset_manager",
            "training/stam",
            "training/explainability",
        ]
    )


class KaggleConfig(BaseModel):
    """Validated ``training/config/kaggle.yaml``."""

    kaggle: KaggleRuntimeSection = Field(default_factory=KaggleRuntimeSection)
    runtime: KaggleRuntimePaths = Field(default_factory=KaggleRuntimePaths)
    dataset_manager: KaggleDatasetManagerSection = Field(
        default_factory=KaggleDatasetManagerSection
    )
    outputs: KaggleOutputs = Field(default_factory=KaggleOutputs)
    install: KaggleInstall = Field(default_factory=KaggleInstall)


class LoggingConfig(BaseModel):
    """Validated ``training/config/logging.yaml`` (Training Logger)."""

    level: str = "INFO"
    dir: str | None = "training/artifacts/logs"
    max_bytes: int = 10 * 1024 * 1024
    backup_count: int = 5
    json_format: bool = True
    console: bool = True


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #

def load_yaml_document(path: Path) -> dict[str, Any]:
    """Read a YAML mapping, raising :class:`ValueError` when malformed."""
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        try:
            raw = yaml.safe_load(fh)
        except yaml.YAMLError as exc:  # pragma: no cover - defensive
            raise ValueError(f"malformed YAML in {path}: {exc}") from exc
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return raw


def _resolve_relative(path: Path, base: Path) -> Path:
    if path.is_absolute():
        return path
    return (base / path).resolve()


def load_paths_config(
    config_path: str | Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> PathsConfig:
    """Load ``paths.yaml`` honouring ``extends`` chains + ``KAGGLE_*`` env vars.

    Args:
        config_path: Optional paths file (defaults to
            ``training/config/paths.yaml`` relative to the repository root).
        env: Environment mapping; defaults to ``os.environ``.

    The ``extends`` key names a parent paths file resolved relative to the
    child file; the parent is merged under the child (child wins) and the
    result is merged under ``KAGGLE_<SECTION>__<KEY>`` overrides.
    """
    if config_path is None:
        config_path = _repo_default(DEFAULT_PATHS_CONFIG)
    path = Path(config_path).resolve()

    raw = load_yaml_document(path)
    merged = _resolve_extends(raw, path)
    overrides = parse_env(env or {}, ENV_PREFIX)
    merged = deep_merge(merged, overrides)
    return PathsConfig.model_validate(merged)


def _resolve_extends(raw: dict[str, Any], path: Path) -> dict[str, Any]:
    """Merge the ``extends`` chain (nearest ancestor first, child wins)."""
    inherited: dict[str, Any] = {}
    parent_ref = raw.get("extends")
    if parent_ref:
        parent_path = _resolve_relative(Path(str(parent_ref)), path.parent)
        parent_raw = load_yaml_document(parent_path)
        inherited = _resolve_extends(parent_raw, parent_path)
    child = {k: v for k, v in raw.items() if k != "extends"}
    return deep_merge(inherited, child)


def load_kaggle_config(
    config_path: str | Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> KaggleConfig:
    """Load and validate ``training/config/kaggle.yaml`` (env overrides win)."""
    if config_path is None:
        config_path = _repo_default("training/config/kaggle.yaml")
    path = Path(config_path).resolve()
    raw = load_yaml_document(path)
    merged = deep_merge(raw, parse_env(env or {}, ENV_PREFIX))
    return KaggleConfig.model_validate(merged)


def load_logging_config(
    config_path: str | Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> LoggingConfig:
    """Load and validate ``training/config/logging.yaml`` (env overrides win)."""
    if config_path is None:
        config_path = _repo_default("training/config/logging.yaml")
    path = Path(config_path).resolve()
    raw = load_yaml_document(path)
    merged = deep_merge(raw.get("logging", raw), parse_env(env or {}, ENV_PREFIX))
    return LoggingConfig.model_validate(merged)


def _repo_default(relative: str) -> Path:
    """Resolve a repository-relative config path (repo root is 3 levels up)."""
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / relative


# --------------------------------------------------------------------------- #
# Workspace layout
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, slots=True)
class WorkspaceLayout:
    """Concrete resolved Kaggle workspace directories."""

    repo_root: Path
    root: Path
    logs: Path
    outputs: Path
    checkpoints: Path
    cache: Path
    configs: Path

    @classmethod
    def resolve(
        cls, paths: PathsConfig, repo_root: Path | None = None, **overrides: str
    ) -> "WorkspaceLayout":
        """Resolve the workspace against a repository root (default: auto).

        Any directory can be overridden with a keyword arg (e.g.
        ``logs=Path("/kaggle/working/logs")``) for Kaggle / testing.
        """
        repo = Path(repo_root or Path(__file__).resolve().parents[2]).resolve()
        ws = paths.workspace
        root = repo / ws.root

        def _dir(name: str, key: str) -> Path:
            value = overrides.get(key)
            if value:
                p = Path(value)
                return p if p.is_absolute() else root / p
            return root / Path(getattr(ws, name)).name

        return cls(
            repo_root=repo,
            root=root,
            logs=_dir("logs_dir", "logs"),
            outputs=_dir("outputs_dir", "outputs"),
            checkpoints=_dir("checkpoints_dir", "checkpoints"),
            cache=_dir("cache_dir", "cache"),
            configs=_dir("configs_dir", "configs"),
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "repo_root": str(self.repo_root),
            "root": str(self.root),
            "logs": str(self.logs),
            "outputs": str(self.outputs),
            "checkpoints": str(self.checkpoints),
            "cache": str(self.cache),
            "configs": str(self.configs),
        }
