"""Kaggle Training Infrastructure (R2.1).

Orchestration + infrastructure for the CropFusion Training Platform running on
Kaggle notebooks / GPU machines. This package wires configuration, environment
detection, logging, workspace, checkpoint, cache, validation and reporting —
it performs **no training logic** (that lives in ``training.training`` and the
sibling packages, wired in a later phase).

Components:

* :mod:`config`       — validated platform config (paths / kaggle / logging)
                        with ``KAGGLE_*`` env overrides + ``extends`` chains.
* :mod:`environment`  — Environment Manager (runtime / system / GPU / deps).
* :mod:`logging`      — Training Logger (JSON / console / rotating + startup,
                        system, experiment logs).
* :mod:`workspace`    — Workspace Manager (folders, cache clean, resume,
                        outputs, temp).
* :mod:`checkpoints`  — Checkpoint Manager (metadata, latest/best/resume,
                        versioning — no model saves).
* :mod:`cache`        — Training Cache (metadata / preprocessing / image /
                        statistics / validation buckets).
* :mod:`validation`   — Training Validator (config / python / GPU / deps /
                        folders / permissions / disk / providers).
* :mod:`reports`      — environment / gpu / dependency / storage / workspace /
                        configuration report builders.
"""

from .config import (
    ConfigRegistry,
    EnvironmentRequirements,
    KaggleConfig,
    LoggingConfig,
    PathsConfig,
    WorkspaceConfig,
    WorkspaceLayout,
    load_kaggle_config,
    load_logging_config,
    load_paths_config,
)
from .workspace import WorkspaceManager
from .checkpoints import CheckpointManager, CheckpointEntry
from .cache import TrainingCache
from .validation import TrainingValidator
from .reports import write_reports

__version__ = "0.1.0"

__all__ = [
    # Config
    "PathsConfig",
    "WorkspaceConfig",
    "ConfigRegistry",
    "EnvironmentRequirements",
    "KaggleConfig",
    "LoggingConfig",
    "WorkspaceLayout",
    "load_paths_config",
    "load_kaggle_config",
    "load_logging_config",
    # Infrastructure
    "WorkspaceManager",
    "CheckpointManager",
    "CheckpointEntry",
    "TrainingCache",
    "TrainingValidator",
    "write_reports",
]
