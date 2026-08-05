"""Version metadata types (Dataset / Model / Inference / Application)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar

from ..enums import ReleaseStatus
from .semver import SemanticVersion


@dataclass(frozen=True, slots=True)
class VersionInfo:
    """A single versioned artifact identifier."""

    #: Unique name of the artifact.
    name: str
    #: Semantic version string (``MAJOR.MINOR.PATCH``).
    version: str
    #: Release lifecycle status.
    status: ReleaseStatus = ReleaseStatus.DRAFT
    #: Optional integrity checksum of the artifact.
    checksum: str | None = None
    #: Optional human readable note.
    message: str = ""
    created_at: datetime = field(default_factory=datetime.now)

    #: Stable kind of the artifact (overridden by subclasses).
    kind: ClassVar[str] = "unknown"

    def __post_init__(self) -> None:
        SemanticVersion.from_string(self.version)

    @property
    def semantic(self) -> SemanticVersion:
        """Parsed :class:`SemanticVersion`."""
        return SemanticVersion.from_string(self.version)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "version": self.version,
            "status": self.status.value,
            "checksum": self.checksum,
            "message": self.message,
            "created_at": self.created_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class DatasetVersion(VersionInfo):
    """Version of a dataset (kind fixed to ``"dataset"``)."""

    kind: ClassVar[str] = "dataset"


@dataclass(frozen=True, slots=True)
class ModelVersion(VersionInfo):
    """Version of a trained model artifact (kind fixed to ``"model"``)."""

    kind: ClassVar[str] = "model"


@dataclass(frozen=True, slots=True)
class InferenceVersion(VersionInfo):
    """Version of the inference/serving runtime (kind fixed to ``"inference"``)."""

    kind: ClassVar[str] = "inference"


@dataclass(frozen=True, slots=True)
class ApplicationVersion(VersionInfo):
    """Version of an application release (kind fixed to ``"application"``)."""

    kind: ClassVar[str] = "application"
