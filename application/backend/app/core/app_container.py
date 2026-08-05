"""Application container — the composition root.

Wires the Configuration / Repository / Service / Model containers and exposes
them on ``app.state``. Heavy components (Dataset Manager, STAM, model) are built
lazily in :meth:`initialize`, so tests can override providers first.
"""

from __future__ import annotations

from typing import Any

from app.core.config import Settings, load_settings
from app.core.container import Container
from app.core.database import Database
from app.core.logging import get_logger
from app.modules.dataset.service import DatasetService
from app.modules.explainability.service import ExplainabilityService
from app.modules.gis.service import GISService, Location
from app.modules.health.service import HealthService
from app.modules.inference.service import InferenceEngine
from app.modules.monitoring.service import MonitoringService
from app.services.cache import build_cache
from app.services.metrics import MetricsRegistry
from app.services.model_registry import ModelRegistry
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
        self.services.register(
            "cache", lambda: build_cache(self._settings.cache), singleton=True
        )
        self.services.register(
            "rate_limiter",
            lambda: build_rate_limiter(self._settings.rate_limit),
            singleton=True,
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
                self.model.resolve("dataset_service"),
                self.repositories.resolve("database"),
            ),
            singleton=True,
        )
        from database.services.redis_store import build_redis_store

        self.services.register(
            "redis_store",
            lambda: build_redis_store(self._settings.redis),
            singleton=True,
        )

    def _register_repositories(self) -> None:
        self.repositories.register(
            "database", lambda: Database(self._settings.database), singleton=True
        )

    def _register_model_providers(self) -> None:
        self.model.register(
            "dataset_manager", lambda: self._build_dataset_manager(), singleton=True
        )
        self.model.register("stam", lambda: self._build_stam(), singleton=True)
        self.model.register(
            "preprocessor", lambda: self._build_preprocessor(), singleton=True
        )
        self.model.register(
            "model_registry",
            lambda: ModelRegistry(self._settings.model),
            singleton=True,
        )
        self.model.register(
            "dataset_service",
            lambda: DatasetService(
                self.model.resolve("dataset_manager"), self._settings.dataset
            ),
            singleton=True,
        )
        self.model.register(
            "gis_service", lambda: self._build_gis_service(), singleton=True
        )
        self.model.register(
            "inference_engine",
            lambda: InferenceEngine(
                self.model.resolve("model_registry"),
                stam=self.model.resolve("stam"),
                preprocessor=self.model.resolve("preprocessor"),
                config=self._settings.inference,
                cache=self.services.resolve("cache"),
            ),
            singleton=True,
        )
        self.model.register(
            "explainability_service",
            lambda: self._build_explainability_service(),
            singleton=True,
        )

    # ------------------------------------------------------------------ #
    # Startup
    # ------------------------------------------------------------------ #

    async def initialize(self) -> None:
        """Build heavy components: database, dataset manager, STAM, model."""
        with _startup_timer("database"):
            database = self.repositories.resolve("database")
            await database.connect()
            await database.create_all()
            await self._seed_if_configured(database)

        with _startup_timer("dataset"):
            self.model.resolve("dataset_manager")
            self.model.resolve("stam")
            self.model.resolve("dataset_service")

        with _startup_timer("model"):
            registry = self.model.resolve("model_registry")
            try:
                registry.load()
                registry.warmup()
            except Exception as exc:
                logger.warning("model not loaded at startup ({})", exc)

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
                    session,
                    include_boundaries=seed.include_boundaries,
                    csv_path=seed.csv_path,
                )
                logger.info("database seeded", **summary)
            except Exception as exc:  # pragma: no cover - best effort
                logger.warning("startup seeding failed ({})", exc)

    # ------------------------------------------------------------------ #
    # Heavy builders
    # ------------------------------------------------------------------ #

    def _build_dataset_manager(self) -> Any:
        from training.dataset_manager import DatasetManager, Settings as DMSettings

        ds = self._settings.dataset
        return DatasetManager(
            DMSettings(
                dataset_root=ds.dataset_root or "datasets",
                catalog_name=ds.catalog_name,
                admin_dir=ds.admin_dir,
                logging={"console": False, "level": "ERROR"},
            )
        )

    def _build_stam(self) -> Any:
        from training.stam import STAM, StamConfig

        manager = self.model.resolve("dataset_manager")
        ds = self._settings.dataset
        return STAM(
            manager,
            StamConfig(
                patch={"size": ds.patch_size},
                tabular={
                    "table": "crop_yield.csv",
                    "village_column": "village",
                    "district_column": "district",
                    "year_column": "year",
                    "season_column": "season",
                    "crop_column": "crop",
                    "yield_column": "yield_kg",
                },
                admin={
                    "boundaries": [f"raw/{ds.catalog_name}/boundaries.geojson"],
                    "name_column": "name",
                    "level_column": "level",
                },
                image={"resolution": ds.image_resolution, "require_pairs": ds.require_pairs},
            ),
        )

    def _build_preprocessor(self) -> Any:
        from training.preprocessing import Preprocessor

        preprocessor_dir = self._settings.model.preprocessor_dir
        if preprocessor_dir:
            try:
                return Preprocessor.load(preprocessor_dir)
            except Exception as exc:
                logger.warning("preprocessor load failed ({}); using empty", exc)
        return Preprocessor()

    def _build_gis_service(self) -> GISService:
        stam = self.model.resolve("stam")
        locations = self._discover_locations(stam)
        return GISService(locations=locations)

    def _build_explainability_service(self) -> ExplainabilityService:
        from training.explainability import Explainer as AIExplainer
        from training.explainability.config import ExplainabilityConfig

        registry = self.model.resolve("model_registry")
        preprocessor = self.model.resolve("preprocessor")
        stam = self.model.resolve("stam")
        ai_explainer = AIExplainer(
            registry.model,
            preprocessor,
            ExplainabilityConfig(),
            observations=None,
            extractor=getattr(stam, "get_patch", None),
        )
        return ExplainabilityService(ai_explainer, self._settings, stam)

    def _discover_locations(self, stam: Any) -> list[Location]:
        """Enumerate dataset locations from the STAM metadata (best-effort)."""
        locations: list[Location] = []
        try:
            manager = self.model.resolve("dataset_manager")
            rows = manager.list_locations() if hasattr(manager, "list_locations") else []
            for row in rows[:500]:
                locations.append(
                    Location(
                        id=str(row.get("id", row.get("village", len(locations)))),
                        lon=float(row.get("lon", 0.0)),
                        lat=float(row.get("lat", 0.0)),
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
