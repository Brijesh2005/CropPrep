"""Training Logger for the Kaggle Training Platform (R2.1).

Wraps ``shared.logging.setup_logging`` (JSON files, compact console, rotating
handlers) and adds three dedicated child loggers under ``cropfusion.training``:

* ``startup``   — ``startup.log``      (bootstrap / environment readiness)
* ``system``    — ``system.log``       (workspace / cache / checkpoint events)
* ``experiment``— ``experiment.log``   (training-run orchestration events)

Each child logger gets its own rotating JSON file handler and propagates to the
parent console. Pure infrastructure — no training logic.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from shared.logging import JsonFormatter, get_logger, log_dict, setup_logging

from .config import LoggingConfig

#: Marker attribute shared with ``shared.logging`` for handler idempotency.
_MARKER = "_cf_owned"

#: Child logger names + rotating file names.
_CHILDREN = {
    "startup": "startup.log",
    "system": "system.log",
    "experiment": "experiment.log",
}

NAMESPACE = "cropfusion.training"


class TrainingLogger:
    """Structured logger for the Training Platform.

    Args:
        config: Optional :class:`LoggingConfig` (defaults to sensible values
            matching ``training/config/logging.yaml``).
        log_dir: Directory for log files. When omitted, resolves from
            ``config.dir`` (repository-relative) falling back to
            ``training/kaggle/logs``.
    """

    def __init__(
        self,
        config: LoggingConfig | None = None,
        log_dir: str | Path | None = None,
    ) -> None:
        self.config = config or LoggingConfig()
        if log_dir is not None:
            self.log_dir = Path(log_dir)
        elif self.config.dir:
            self.log_dir = Path(__file__).resolve().parents[2] / self.config.dir
        else:
            self.log_dir = Path(__file__).resolve().parents[0] / "logs"
        self._children: dict[str, logging.Logger] = {}
        self._configured = False

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #

    def setup(self) -> "TrainingLogger":
        """Configure the platform logger hierarchy (idempotent)."""
        if self._configured:
            return self
        setup_logging(
            level=self.config.level,
            log_dir=self.log_dir,
            max_bytes=self.config.max_bytes,
            backup_count=self.config.backup_count,
            output="json" if self.config.json_format else "compact",
            console=self.config.console,
            console_output="compact",
            profile="training",
            logger_name=NAMESPACE,
        )
        for name, file_name in _CHILDREN.items():
            self._children[name] = self._attach_child(name, file_name)
        self._configured = True
        return self

    def _attach_child(self, name: str, file_name: str) -> logging.Logger:
        logger = get_logger(f"{NAMESPACE}.{name}")
        logger.setLevel(logging.getLevelName(self.config.level))
        logger.propagate = True
        for handler in list(logger.handlers):
            if getattr(handler, _MARKER, False):
                logger.removeHandler(handler)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            self.log_dir / file_name,
            maxBytes=self.config.max_bytes,
            backupCount=self.config.backup_count,
            encoding="utf-8",
        )
        handler.setFormatter(JsonFormatter())
        handler.setLevel(logging.getLevelName(self.config.level))
        setattr(handler, _MARKER, True)
        logger.addHandler(handler)
        return logger

    # ------------------------------------------------------------------ #
    # Emitters
    # ------------------------------------------------------------------ #

    def log_startup(self, event: str, **fields: Any) -> None:
        """Write a structured line to ``startup.log``."""
        self.setup()
        log_dict(self._children["startup"], logging.INFO, event, **fields)

    def log_system(self, event: str, **fields: Any) -> None:
        """Write a structured line to ``system.log``."""
        self.setup()
        log_dict(self._children["system"], logging.INFO, event, **fields)

    def log_experiment(self, event: str, **fields: Any) -> None:
        """Write a structured line to ``experiment.log``."""
        self.setup()
        log_dict(self._children["experiment"], logging.INFO, event, **fields)

    def child(self, name: str) -> logging.Logger:
        """Return a configured child logger (raises for unknown names)."""
        if name not in _CHILDREN:
            raise KeyError(f"unknown training log: {name!r} (got {self._children})")
        self.setup()
        return self._children[name]

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    def log_files(self) -> dict[str, str]:
        """Map of log name → absolute path (post-setup)."""
        return {
            name: str(self.log_dir / file_name) for name, file_name in _CHILDREN.items()
        }

    @property
    def configured(self) -> bool:
        return self._configured
