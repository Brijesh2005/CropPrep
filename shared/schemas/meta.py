"""Config / training / version / release metadata schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..enums import ModelStatus, ReleaseStatus, TrainingStage


@dataclass(slots=True)
class ConfigMetadataSchema:
    """Metadata about a loaded configuration."""

    source: str | None = None
    env_prefix: str | None = None
    loaded_at: datetime = field(default_factory=datetime.now)
    settings_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "env_prefix": self.env_prefix,
            "loaded_at": self.loaded_at.isoformat(),
            "settings_hash": self.settings_hash,
        }


@dataclass(slots=True)
class TrainingRunSchema:
    """Metadata about a training run."""

    run_id: str
    model_name: str
    stage: TrainingStage = TrainingStage.INIT
    status: ModelStatus = ModelStatus.PENDING
    epoch: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    dataset_version: str = "0.0.0"
    started_at: datetime = field(default_factory=datetime.now)
    finished_at: datetime | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "model_name": self.model_name,
            "stage": self.stage.value,
            "status": self.status.value,
            "epoch": self.epoch,
            "metrics": self.metrics,
            "dataset_version": self.dataset_version,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "extra": self.extra,
        }


@dataclass(slots=True)
class ReleaseMetadataSchema:
    """Metadata about a versioned release."""

    kind: str
    name: str
    version: str
    status: ReleaseStatus = ReleaseStatus.DRAFT
    changelog: str = ""
    released_at: datetime = field(default_factory=datetime.now)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "version": self.version,
            "status": self.status.value,
            "changelog": self.changelog,
            "released_at": self.released_at.isoformat(),
            "extra": self.extra,
        }
