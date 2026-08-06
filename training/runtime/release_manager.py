"""Release manager (Phase R6).

:class:`ReleaseManager` is the single owner of the releases root directory. It
discovers, validates, activates and rolls back releases and persists the
activation state so a runtime can resume its previous release after restart.

Release lifecycle::

    discover() -> list[ReleaseInfo]      scan the releases root
    validate(version) -> result         full validation battery
    activate(version)                    set the active release
    rollback()                           restore the previous release
    current_version() / active()         read the state
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from shared.versioning import SemanticVersion

from .config import RuntimeConfig
from .exceptions import (
    ReleaseActivationError,
    ReleaseNotFoundError,
    ReleaseRollbackError,
    ReleaseValidationError,
)
from .layout import (
    ReleaseInfo,
    ReleaseLayout,
    ReleaseManifest,
    iter_release_dirs,
    release_dir_name,
    resolve_release,
)
from .validation import ReleaseValidationResult, ReleaseValidator

#: Name of the runtime state file inside the releases root.
STATE_FILE = "runtime_state.json"


@dataclass
class RuntimeState:
    """Persisted activation state."""

    active: str | None
    history: list[str]
    updated_at: str

    @classmethod
    def empty(cls) -> "RuntimeState":
        return cls(active=None, history=[], updated_at=_now())

    def to_dict(self) -> dict[str, Any]:
        return {
            "active": self.active,
            "history": list(self.history),
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RuntimeState":
        return cls(
            active=payload.get("active"),
            history=list(payload.get("history") or []),
            updated_at=str(payload.get("updated_at") or _now()),
        )


class ReleaseManager:
    """Discover / validate / activate / rollback release packages.

    Args:
        config: Validated :class:`RuntimeConfig` (``None`` = defaults). The
            releases root comes from ``general.releases_root``.
        releases_root: Explicit override for the releases root directory.
    """

    def __init__(
        self,
        config: RuntimeConfig | None = None,
        *,
        releases_root: str | Path | None = None,
    ) -> None:
        self.config = config or RuntimeConfig()
        self.releases_root = Path(
            releases_root or self.config.general.releases_root
        )
        self.state_path = self.releases_root / STATE_FILE
        self.validator = ReleaseValidator(self.config)
        self._state = RuntimeState.empty()

    # ------------------------------------------------------------------ #
    # Initialisation / state
    # ------------------------------------------------------------------ #

    def initialize(self) -> "ReleaseManager":
        """Create the releases root (if missing) and load persisted state."""
        self.releases_root.mkdir(parents=True, exist_ok=True)
        if self.state_path.exists():
            self._state = RuntimeState.from_dict(_load_json(self.state_path))
        return self

    def state(self) -> RuntimeState:
        return self._state

    def _save_state(self) -> None:
        self._state.updated_at = _now()
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(self._state.to_dict(), indent=2), encoding="utf-8"
        )

    # ------------------------------------------------------------------ #
    # Discovery
    # ------------------------------------------------------------------ #

    def discover(self) -> list[ReleaseInfo]:
        """Scan the releases root and return every release, newest first."""
        found: list[ReleaseInfo] = []
        for name, path, parsed in iter_release_dirs(self.releases_root):
            manifest: ReleaseManifest | None = None
            layout = ReleaseLayout(path)
            try:
                manifest = layout.manifest()
            except Exception:  # ReleaseLayoutError
                pass
            version = parsed
            if version is None and manifest is not None:
                version = manifest.package_version
            if version is None:
                continue
            try:
                SemanticVersion.from_string(version)
            except Exception:  # InvalidVersionError
                continue
            _, structure_errors = layout.is_valid_structure()
            info = ReleaseInfo(
                name=name,
                version=version,
                path=path,
                manifest=manifest,
                model_version=(
                    manifest.model_version if manifest else None
                ),
                dataset_version=(
                    manifest.dataset_version if manifest else None
                ),
                formats=layout.formats,
                structure_errors=structure_errors,
            )
            found.append(info)
        found.sort(key=lambda info: SemanticVersion.from_string(info.version), reverse=True)
        return found

    def versions(self) -> list[str]:
        return [info.version for info in self.discover()]

    def get(self, version: str) -> ReleaseInfo:
        """Return the :class:`ReleaseInfo` for ``version``.

        Raises:
            ReleaseNotFoundError: When the version does not exist.
        """
        for info in self.discover():
            if info.version == version:
                return info
        raise ReleaseNotFoundError(
            f"release {version!r} not found",
            detail={"releases_root": str(self.releases_root), "version": version},
        )

    def release_path(self, version: str) -> Path:
        """Resolve the release package directory for ``version``."""
        try:
            return resolve_release(self.releases_root, version)
        except ReleaseNotFoundError:
            raise

    def latest(self) -> ReleaseInfo:
        """The newest release (highest semantic version)."""
        releases = self.discover()
        if not releases:
            raise ReleaseNotFoundError(
                "no releases found",
                detail={"releases_root": str(self.releases_root)},
            )
        return releases[0]

    def current_version(self) -> str | None:
        """The persisted active version (no directory access)."""
        return self._state.active

    def active(self) -> ReleaseInfo | None:
        """The currently active release (or ``None``)."""
        version = self._state.active
        if version is None:
            return None
        try:
            return self.get(version)
        except ReleaseNotFoundError:
            return None

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def validate(
        self, version: str | None = None, *, strict: bool | None = None
    ) -> ReleaseValidationResult:
        """Validate a release (defaults to the active one).

        Raises:
            ReleaseNotFoundError: When the version does not exist.
            ReleaseValidationError: When strict validation fails.
        """
        version = version or self._state.active or self.latest().version
        info = self.get(version)
        return self.validator.validate_release(info.path, strict=strict)

    # ------------------------------------------------------------------ #
    # Activation / rollback
    # ------------------------------------------------------------------ #

    def activate(
        self, version: str | None = None, *, validate: bool = True
    ) -> ReleaseInfo:
        """Activate a release (defaults to the latest).

        Args:
            version: Release version to activate. When omitted, the latest
                discovered release is activated.
            validate: Run the validation battery first (strict mode raises).

        Raises:
            ReleaseNotFoundError: When the version does not exist.
            ReleaseValidationError: When strict validation fails.
        """
        version = version or self.latest().version
        info = self.get(version)
        if validate:
            result = self.validator.validate_release(info.path)
            if not result.valid:
                raise ReleaseValidationError(
                    f"cannot activate release {version}: validation failed",
                    detail=result.to_dict(),
                )
        previous = self._state.active
        if previous != version:
            if previous is not None:
                self._state.history.append(previous)
            self._state.active = version
            self._save_state()
        return info

    def rollback(self, *, validate: bool = True) -> ReleaseInfo:
        """Revert to the previously active release.

        Raises:
            ReleaseRollbackError: When there is no previous release to restore.
        """
        if not self._state.history:
            raise ReleaseRollbackError(
                "no previous release available for rollback",
                detail={"state": self._state.to_dict()},
            )
        previous = self._state.history.pop()
        info = self.get(previous)
        if validate:
            result = self.validator.validate_release(info.path)
            if not result.valid:
                self._state.history.append(previous)
                self._save_state()
                raise ReleaseValidationError(
                    f"cannot roll back to release {previous}: validation failed",
                    detail=result.to_dict(),
                )
        self._state.active = previous
        self._save_state()
        return info

    # ------------------------------------------------------------------ #
    # Status
    # ------------------------------------------------------------------ #

    def status(self) -> dict[str, Any]:
        """Overview of every release plus the active version."""
        releases = self.discover()
        return {
            "releases_root": str(self.releases_root),
            "active": self._state.active,
            "history": list(self._state.history),
            "releases": [info.to_dict() for info in releases],
        }

    def list(self) -> list[ReleaseInfo]:
        return self.discover()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    import json

    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}
