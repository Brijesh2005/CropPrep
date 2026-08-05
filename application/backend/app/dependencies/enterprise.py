"""Phase 10 enterprise dependency factories.

Builds the Phase 10 services per-request, bound to the request's database
session and the application's Redis store. Mirrors the Phase 8 pattern of
``app/modules/*/dependencies.py``.
"""

from __future__ import annotations

from typing import Any

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies.container import get_container
from app.dependencies.database import get_session
from database.repositories import (
    AdministrativeBoundaryRepository,
    AppConfigurationRepository,
    AuditLogRepository,
    CropRepository,
    DatasetVersionRepository,
    FeedbackRepository,
    ModelVersionRepository,
    NotificationRepository,
    PredictionMetadataRepository,
    PredictionRepository,
    ResearchExperimentRepository,
    SeasonRepository,
    SpatialLocationRepository,
    SystemLogRepository,
    UserLocationRepository,
    UserPreferenceRepository,
    UserRepository,
    UserSessionRepository,
)
from database.services.redis_store import RedisStore
from database.services import (
    AnalyticsService,
    AuditService,
    CatalogService,
    ConfigService,
    ExperimentService,
    FeedbackService,
    NotificationService,
    PredictionHistoryService,
    ProfileService,
    RegistryService,
    SpatialService,
)
from database.services.auth import AuthService


def get_redis_store(container: Any = Depends(get_container)) -> RedisStore:
    """Resolve the shared Redis (or in-memory) store from the container."""
    return container.services.resolve("redis_store")


def get_enterprise_auth_service(
    session: AsyncSession = Depends(get_session),
    container: Any = Depends(get_container),
) -> AuthService:
    settings = container.config.resolve("settings")
    store = container.services.resolve("redis_store")
    return AuthService(session, settings.security, settings.security.password_policy, store)


def get_profile_service(
    session: AsyncSession = Depends(get_session),
) -> ProfileService:
    return ProfileService(
        UserRepository(session),
        UserPreferenceRepository(session),
        UserLocationRepository(session),
    )


def get_prediction_history_service(
    session: AsyncSession = Depends(get_session),
) -> PredictionHistoryService:
    return PredictionHistoryService(
        PredictionRepository(session), PredictionMetadataRepository(session)
    )


def get_analytics_service(
    session: AsyncSession = Depends(get_session),
    store: RedisStore = Depends(get_redis_store),
) -> AnalyticsService:
    return AnalyticsService(
        PredictionRepository(session),
        UserRepository(session),
        FeedbackRepository(session),
        NotificationRepository(session),
        store,
    )


def get_notification_service(
    session: AsyncSession = Depends(get_session),
    store: RedisStore = Depends(get_redis_store),
) -> NotificationService:
    return NotificationService(NotificationRepository(session), store)


def get_feedback_service(
    session: AsyncSession = Depends(get_session),
) -> FeedbackService:
    return FeedbackService(FeedbackRepository(session))


def get_audit_service(
    session: AsyncSession = Depends(get_session),
) -> AuditService:
    return AuditService(AuditLogRepository(session), SystemLogRepository(session))


def get_registry_service(
    session: AsyncSession = Depends(get_session),
) -> RegistryService:
    return RegistryService(
        ModelVersionRepository(session), DatasetVersionRepository(session)
    )


def get_catalog_service(
    session: AsyncSession = Depends(get_session),
) -> CatalogService:
    return CatalogService(CropRepository(session), SeasonRepository(session))


def get_spatial_service(
    session: AsyncSession = Depends(get_session),
    store: RedisStore = Depends(get_redis_store),
) -> SpatialService:
    return SpatialService(
        SpatialLocationRepository(session),
        AdministrativeBoundaryRepository(session),
        store,
    )


def get_experiment_service(
    session: AsyncSession = Depends(get_session),
) -> ExperimentService:
    return ExperimentService(ResearchExperimentRepository(session))


def get_config_service(
    session: AsyncSession = Depends(get_session),
) -> ConfigService:
    return ConfigService(AppConfigurationRepository(session))


def get_user_sessions_service(
    session: AsyncSession = Depends(get_session),
    container: Any = Depends(get_container),
) -> tuple[UserSessionRepository, Any]:
    """Session listing / revocation support (repository + security settings)."""
    settings = container.config.resolve("settings")
    return UserSessionRepository(session), settings.security
