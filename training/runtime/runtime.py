"""Inference runtime (Phase R6).

:class:`InferenceRuntime` is the orchestrator that initialises the runtime
environment, loads a release, validates it, warms the model up, monitors
health and shutdown, and supports hot reload + rollback.

Lifecycle::

    runtime = InferenceRuntime(config)
    runtime.start(version="1.0.0")     # init -> validate -> activate -> load -> warmup
    runtime.health().to_dict()          # readiness snapshot
    runtime.poll_reload()               # hot reload (poll driven)
    runtime.rollback()                  # revert to the previous release
    runtime.shutdown()                  # release resources

The runtime never imports the Kaggle catalog, the dataset manager or the
training loop — it consumes only the exported artefacts of a release package.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from .cache import RuntimeCache
from .config import RuntimeConfig
from .exceptions import (
    MemoryLimitError,
    ReleaseNotFoundError,
    RuntimeEnvironmentError,
)
from .health import (
    STATUS_DEGRADED,
    STATUS_LOADING,
    STATUS_NOT_READY,
    STATUS_READY,
    HealthReport,
    MemoryMonitor,
)
from .layout import ReleaseInfo, ReleaseLayout
from .metadata_loader import MetadataLoader
from .model_loader import ModelLoader
from .preprocess_loader import PreprocessLoader
from .release_manager import ReleaseManager
from .validation import ReleaseValidationResult, ReleaseValidator


class InferenceRuntime:
    """Production-grade runtime for a CropFusion release package.

    Args:
        config: Validated :class:`RuntimeConfig` (``None`` = defaults).
    """

    def __init__(self, config: RuntimeConfig | None = None) -> None:
        self.config = config or RuntimeConfig()
        cache_enabled = self.config.cache.enabled
        self.cache = RuntimeCache(
            max_bytes=self.config.cache.max_bytes if cache_enabled else 0,
            max_entries=self.config.cache.max_entries if cache_enabled else 0,
            ttl_seconds=self.config.cache.ttl_seconds,
        )
        self.manager = ReleaseManager(self.config)
        self.validator = ReleaseValidator(self.config)
        self.memory = MemoryMonitor(self.config.memory, self.cache)

        self.model_loader: ModelLoader | None = None
        self.preprocess_loader: PreprocessLoader | None = None
        self.metadata_loader: MetadataLoader | None = None

        self._started = False
        self._ready = False
        self._startup_time_ms: float | None = None
        self._started_at: float | None = None
        self._hot_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._reloads = 0
        self._snapshot: dict[str, Any] = {}
        self._last_validation: ReleaseValidationResult | None = None

    # ------------------------------------------------------------------ #
    # Initialisation
    # ------------------------------------------------------------------ #

    def initialize(self) -> "InferenceRuntime":
        """Create the releases root, load state and snapshot the root."""
        self.manager.initialize()
        self._started_at = time.monotonic()
        self._started = True
        self._snapshot = self._release_snapshot()
        return self

    def start(
        self,
        version: str | None = None,
        *,
        validate: bool | None = None,
    ) -> "InferenceRuntime":
        """Initialise, activate and load a release (defaults to latest)."""
        self.initialize()
        self.load_release(version, validate=validate)
        return self

    # ------------------------------------------------------------------ #
    # Release lifecycle
    # ------------------------------------------------------------------ #

    def load_release(
        self,
        version: str | None = None,
        *,
        validate: bool | None = None,
    ) -> ReleaseInfo:
        """Activate + load + warm up a release (defaults to the latest).

        Args:
            version: Release version to load. When omitted, the persisted
                active version is used (falling back to the latest).
            validate: Run the full validation battery first (defaults to
                ``True``; strictness follows ``validation.strict``).

        Raises:
            ReleaseValidationError: When strict validation fails.
            ModelLoadError / PreprocessLoadError / MetadataLoadError: When a
                component cannot be loaded.
        """
        if validate is None:
            validate = True
        version = version or self.manager.current_version()
        info = self.manager.activate(version, validate=validate)
        self._load_release_info(info)
        return info

    def reload_release(self, version: str | None = None) -> ReleaseInfo:
        """Re-read and reload the active release from disk (hot reload).

        The activation state is left untouched; the components (model /
        preprocess / metadata) are rebuilt from the release directory.
        """
        version = version or self.manager.current_version() or self.manager.latest().version
        info = self.manager.get(version)
        if self.config.validation.verify_checksums or self.config.validation.strict:
            self._last_validation = self.validator.validate_release(info.path)
        self._load_release_info(info)
        return info

    def rollback(self) -> ReleaseInfo:
        """Revert to the previously active release and reload it."""
        info = self.manager.rollback()
        self._load_release_info(info)
        return info

    def validate(self, version: str | None = None) -> ReleaseValidationResult:
        """Validate a release (defaults to the active / latest one)."""
        if version is None:
            version = self.manager.current_version() or self.manager.latest().version
        info = self.manager.get(version)
        self._last_validation = self.validator.validate_release(info.path)
        return self._last_validation

    # ------------------------------------------------------------------ #
    # Component loading
    # ------------------------------------------------------------------ #

    def _load_release_info(self, info: ReleaseInfo) -> None:
        self._unload_components()
        self._ready = False
        layout = ReleaseLayout(info.path)
        try:
            self.model_loader = ModelLoader(layout, self.config)
            self.model_loader.load()
            if self.config.preprocess.required:
                self.preprocess_loader = PreprocessLoader(layout, self.config)
                self.preprocess_loader.load()
            if self.config.metadata.required:
                self.metadata_loader = MetadataLoader(
                    layout, self.config, self.cache
                )
                self.metadata_loader.load()
            self.warmup()
            self.memory.check()
        except (MemoryLimitError, RuntimeEnvironmentError):
            self._unload_components()
            raise
        if self._startup_time_ms is None:
            self._startup_time_ms = _elapsed_ms(self._started_at)
        self._ready = True
        self._snapshot = self._release_snapshot()

    def warmup(self) -> bool:
        """Run the model warm-up (a no-op when the model is not loaded)."""
        if self.model_loader is None or not self.model_loader.health().loaded:
            return False
        return self.model_loader.warmup()

    # ------------------------------------------------------------------ #
    # Health
    # ------------------------------------------------------------------ #

    def health(self) -> HealthReport:
        """The current readiness snapshot.

        Raises:
            HealthError: When health reporting is disabled.
        """
        if not self.config.health.enabled:
            raise RuntimeEnvironmentError(
                "health reporting is disabled (health.enabled=false)"
            )
        model = self.model_loader.health() if self.model_loader else None
        preprocess = (
            self.preprocess_loader.health() if self.preprocess_loader else None
        )
        metadata = (
            self.metadata_loader.health() if self.metadata_loader else None
        )

        release_ready = self.manager.active() is not None and self._ready
        model_ready = bool(model and model.loaded and model.warmup_ok)
        preprocess_ready = bool(preprocess and preprocess.loaded)
        metadata_ready = bool(metadata and metadata.loaded)
        warmup_ok = bool(model and model.warmup_ok)

        checks = {
            "release_ready": release_ready,
            "model_ready": model_ready,
            "preprocess_ready": preprocess_ready,
            "metadata_ready": metadata_ready,
            "warmup_ok": warmup_ok,
        }

        optional_pre = not self.config.preprocess.required
        optional_meta = not self.config.metadata.required
        core_ok = release_ready and model_ready and warmup_ok
        missing_optional = (
            (not preprocess_ready and optional_pre)
            or (not metadata_ready and optional_meta)
        )

        components_absent = (
            self.model_loader is None
            and self.preprocess_loader is None
            and self.metadata_loader is None
        )
        if not self._started or (not self._ready and components_absent):
            status = STATUS_NOT_READY
            ready = False
        elif core_ok and not missing_optional:
            status = STATUS_READY
            ready = True
        elif core_ok and missing_optional:
            status = STATUS_DEGRADED
            ready = True
        else:
            status = STATUS_LOADING
            ready = False

        memory = self.memory.snapshot()
        return HealthReport(
            status=status,
            ready=ready,
            release_ready=release_ready,
            model_ready=model_ready,
            preprocess_ready=preprocess_ready,
            metadata_ready=metadata_ready,
            version=(
                self.manager.current_version()
                or (self.manager.latest().version if self._started else None)
            ),
            model_version=model.model_version if model else None,
            backend=model.backend if model else None,
            memory=memory,
            startup_time_ms=self._startup_time_ms,
            uptime_seconds=_uptime(self._started_at),
            warmup_ok=warmup_ok,
            checks=checks,
        )

    # ------------------------------------------------------------------ #
    # Hot reload
    # ------------------------------------------------------------------ #

    def start_hot_reload(self, interval: float | None = None) -> threading.Thread:
        """Start a background watcher that polls for release changes.

        Reloads only happen when ``hot_reload.enabled`` is true (default
        behaviour) — set ``RT_HOT_RELOAD__ENABLED=true`` or configure the
        section to turn it on.
        """
        if self._hot_thread is not None:
            return self._hot_thread
        interval = interval or self.config.hot_reload.poll_interval_seconds
        self._stop_event.clear()

        def _loop() -> None:
            while not self._stop_event.wait(interval):
                try:
                    self.poll_reload()
                except Exception:  # noqa: BLE001 - keep the watcher alive
                    continue

        self._hot_thread = threading.Thread(
            target=_loop, name="cropfusion-hot-reload", daemon=True
        )
        self._hot_thread.start()
        return self._hot_thread

    def stop_hot_reload(self) -> None:
        """Stop the background watcher."""
        self._stop_event.set()
        self._hot_thread = None

    def poll_reload(self) -> bool:
        """Check the releases root for changes; reload when configured.

        Returns:
            ``True`` when a change was detected (and, when enabled, the active
            release was reloaded).
        """
        snapshot = self._release_snapshot()
        changed = snapshot != self._snapshot
        self._snapshot = snapshot
        if not changed:
            return False
        self._reloads += 1
        max_reloads = self.config.hot_reload.max_reloads
        if (
            self.config.hot_reload.enabled
            and self.config.hot_reload.auto_reload
            and (max_reloads is None or self._reloads <= max_reloads)
        ):
            self.reload_release()
        return True

    def _release_snapshot(self) -> dict[str, Any]:
        """A fingerprint of the releases root used to detect changes."""
        root = self.manager.releases_root
        entries: dict[str, Any] = {"active": self.manager.current_version()}
        if root.exists():
            for child in sorted(root.iterdir()):
                if not child.is_dir() or not child.name.startswith("cropfusion_release"):
                    continue
                manifest = child / "version" / "manifest.json"
                checksums = child / "version" / "checksums.json"
                entries[child.name] = (
                    _mtime(manifest),
                    _mtime(checksums),
                )
        return entries

    # ------------------------------------------------------------------ #
    # Shutdown / status
    # ------------------------------------------------------------------ #

    def shutdown(self) -> "InferenceRuntime":
        """Stop the watcher, unload every component and reset state."""
        self.stop_hot_reload()
        self._unload_components()
        self.cache.clear()
        self._ready = False
        self._snapshot = {}
        return self

    def _unload_components(self) -> None:
        for loader in (
            self.model_loader,
            self.preprocess_loader,
            self.metadata_loader,
        ):
            if loader is not None:
                try:
                    loader.unload()
                except Exception:  # noqa: BLE001 - best effort
                    pass
        self.model_loader = None
        self.preprocess_loader = None
        self.metadata_loader = None

    def status(self) -> dict[str, Any]:
        """Overview of releases, activation state and readiness."""
        return {
            "started": self._started,
            "ready": self._ready,
            "active": self.manager.current_version(),
            "releases": self.manager.status(),
            "hot_reload": {
                "enabled": self.config.hot_reload.enabled,
                "reloads": self._reloads,
                "watcher_active": self._hot_thread is not None,
            },
            "memory": self.memory.snapshot().to_dict(),
        }


def _mtime(path: Any) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _elapsed_ms(start: float | None) -> float | None:
    if start is None:
        return None
    return (time.monotonic() - start) * 1000.0


def _uptime(start: float | None) -> float:
    if start is None:
        return 0.0
    return time.monotonic() - start
