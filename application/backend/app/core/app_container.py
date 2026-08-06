"""Application container — the composition root.

REPLACES app/core/app_container.py for the Prediction Platform (R6). Compared
to the R1-R5 version:

* ``model_registry`` now comes from ``app.services.release_model_registry``
  (loads only ``cropfusion_release/``) instead of
  ``app.services.model_registry`` (loaded live training checkpoints).
* ``inference_engine`` no longer takes ``stam`` / ``preprocessor`` — the
  release-package engine resolves location + features internally.
* ``dataset_manager``, ``stam`` and the training-time ``preprocessor`` are
  removed entirely — the Prediction Platform must never construct these.
* ``explainability_service`` (which depended on the live training explainer)
  is removed from this composition root; ``PredictionService`` now does a
  lightweight, self-contained explanation (see
  app/modules/predictions/service.py in this batch). Wiring the full
  Explainability module against release-package artifacts is a follow-up.
* GIS locations are now discovered from the release package's
  ``location_index.parquet`` instead of the Dataset Manager.
"""

from __future__ import annotations

from typing import Any

from app.core.config import Settings, load_settings
from app.core.container import Container
from app.core.database import Database
from app.core.logging import get_logger
from app.modules.gis.service import GISService, Location
from app.modules.health.service import HealthService
from app.modules.inference.service import InferenceEngine
from app.modules.monitoring.service import MonitoringService
from app.services.cache import build_cache
from app.services.metrics import MetricsRegistry
from app.services.release_model_registry import ModelRegistry
from app.services.rate_limiter import build_rate_limiter

logger = get_logger("container")


class ApplicationContainer:
    """Composes the DI containers for the whole application."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or load_settings()
        self.config: Container = Container()
        self.repositories: Container = Container()
        self.services: Container = Container()
        self.model: Container = Container()
        self._register_config()
        self._register_services()
        self._register_model_providers()
        self._register_repositories()

    # ------------------------------------------------------------------ #
    # Registration
    # ------------------------------------------------------------------ #

    def _register_config(self) -> None:
        self.config.register("settings", lambda: self._settings, singleton=True)

    def _register_services(self) -> None:
        self.services.register("cache", lambda: build_cache(self._settings.cache), singleton=True)
        self.services.register(
            "rate_limiter", lambda: build_rate_limiter(self._settings.rate_limit), singleton=True
        )
        self.services.register("metrics", lambda: MetricsRegistry(), singleton=True)
        self.services.register(
            "monitoring_service",
            lambda: MonitoringService(self.services.resolve("metrics")),
            singleton=True,
        )
        self.services.register(
            "health_service",
            lambda: HealthService(
                self._settings,
                self.model.resolve("model_registry"),
                # No live dataset manager in the Prediction Platform — the
                # release package itself stands in for "dataset readiness".
                self.model.resolve("model_registry"),
                self.repositories.resolve("database"),
            ),
            singleton=True,
        )
        from database.services.redis_store import build_redis_store

        self.services.register(
            "redis_store", lambda: build_redis_store(self._settings.redis), singleton=True
        )

    def _register_repositories(self) -> None:
        self.repositories.register(
            "database", lambda: Database(self._settings.database), singleton=True
        )

    def _register_model_providers(self) -> None:
        self.model.register(
            "model_registry", lambda: ModelRegistry(self._settings.model), singleton=True
        )
        self.model.register("gis_service", lambda: self._build_gis_service(), singleton=True)
        self.model.register(
            "inference_engine",
            lambda: InferenceEngine(
                self.model.resolve("model_registry"),
                config=self._settings.inference,
                cache=self.services.resolve("cache"),
            ),
            singleton=True,
        )

    # ------------------------------------------------------------------ #
    # Startup
    # ------------------------------------------------------------------ #

    async def initialize(self) -> None:
        """Build heavy components: database, release package, model."""
        with _startup_timer("database"):
            database = self.repositories.resolve("database")
            await database.connect()
            await database.create_all()
            await self._seed_if_configured(database)

        with _startup_timer("model"):
            registry = self.model.resolve("model_registry")
            try:
                registry.load()
                registry.warmup()
            except Exception as exc:
                logger.warning("release package not loaded at startup ({})", exc)

        with _startup_timer("gis"):
            self.model.resolve("gis_service")

        logger.info("application container initialised")

    async def shutdown(self) -> None:
        engine = self.model.resolve("inference_engine")
        await engine.stop()
        database = self.repositories.resolve("database")
        await database.close()
        try:
            store = self.services.resolve("redis_store")
            await store.close()
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("redis store close failed ({})", exc)

    def wire(self, app: Any) -> None:
        """Attach the container to ``app.state`` for the dependencies."""
        app.state.container = self
        app.state.database = self.repositories.resolve("database")

    # ------------------------------------------------------------------ #
    # Seeding
    # ------------------------------------------------------------------ #

    async def _seed_if_configured(self, database: Database) -> None:
        seed = self._settings.seed
        if not seed.on_startup:
            return
        from database.seeds.runner import seed_database

        async with database.session_factory() as session:
            try:
                summary = await seed_database(
                    session, include_boundaries=seed.include_boundaries, csv_path=seed.csv_path
                )
                logger.info("database seeded", **summary)
            except Exception as exc:  # pragma: no cover - best effort
                logger.warning("startup seeding failed ({})", exc)

    # ------------------------------------------------------------------ #
    # GIS — sourced from the release package, not the Dataset Manager
    # ------------------------------------------------------------------ #

    def _build_gis_service(self) -> GISService:
        return GISService(locations=self._discover_locations())

    def _discover_locations(self) -> list[Location]:
        """Enumerate villages from the release package's location_index
        (best-effort; empty if the package hasn't loaded yet)."""
        locations: list[Location] = []
        registry = self.model.resolve("model_registry")
        package = getattr(registry, "package", None)
        if package is None:
            return locations
        try:
            df = package.location_index
            for i, row in df.iterrows():
                locations.append(
                    Location(
                        id=str(row.get("village", i)),
                        lon=float(row["lon"]),
                        lat=float(row["lat"]),
                        name=str(row.get("village", "")),
                        admin={
                            "village": str(row.get("village", "")),
                            "district": str(row.get("district", "")),
                        },
                    )
                )
        except Exception as exc:
            logger.warning("location discovery failed ({})", exc)
        return locations


def _startup_timer(name: str):
    from contextlib import contextmanager

    from app.core.logging import PerformanceTimer

    @contextmanager
    def _ctx():
        timer = PerformanceTimer(f"startup.{name}")
        yield
        timer.stop()

    return _ctx()
