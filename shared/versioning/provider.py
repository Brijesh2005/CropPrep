"""Port for reading and bumping versions of artifacts."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .versions import VersionInfo


class VersionProvider(ABC):
    """Read and bump the version of a named artifact."""

    @abstractmethod
    def current(self, name: str) -> VersionInfo | None:
        """Return the current version of ``name``, or None when absent."""

    @abstractmethod
    def list(self, name: str) -> list[VersionInfo]:
        """Return the version history of ``name``, newest first."""

    @abstractmethod
    def bump(
        self, name: str, part: str = "patch", *, message: str = ""
    ) -> VersionInfo:
        """Bump the current version of ``name`` (major/minor/patch)."""
