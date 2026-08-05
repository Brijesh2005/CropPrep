"""Repository layer (Phase 10)."""

from __future__ import annotations

from database.repositories.access_repository import PermissionRepository, RoleRepository
from database.repositories.base import DataRepository
from database.repositories.catalog_repository import CropRepository, SeasonRepository
from database.repositories.engagement_repository import FeedbackRepository, NotificationRepository
from database.repositories.experiment_repository import (
    AppConfigurationRepository,
    ResearchExperimentRepository,
)
from database.repositories.logging_repository import AuditLogRepository, SystemLogRepository
from database.repositories.metadata_repository import PredictionMetadataRepository
from database.repositories.prediction_repository import PredictionRepository
from database.repositories.profile_repository import UserLocationRepository, UserPreferenceRepository
from database.repositories.registry_repository import DatasetVersionRepository, ModelVersionRepository
from database.repositories.session_repository import UserSessionRepository
from database.repositories.spatial_repository import (
    AdministrativeBoundaryRepository,
    SpatialLocationRepository,
)
from database.repositories.token_repository import (
    EmailVerificationTokenRepository,
    PasswordResetTokenRepository,
    RefreshTokenRepository,
)
from database.repositories.user_repository import UserRepository

__all__ = [
    "AdministrativeBoundaryRepository",
    "AppConfigurationRepository",
    "AuditLogRepository",
    "CropRepository",
    "DataRepository",
    "DatasetVersionRepository",
    "EmailVerificationTokenRepository",
    "FeedbackRepository",
    "ModelVersionRepository",
    "NotificationRepository",
    "PasswordResetTokenRepository",
    "PermissionRepository",
    "PredictionMetadataRepository",
    "PredictionRepository",
    "RefreshTokenRepository",
    "ResearchExperimentRepository",
    "RoleRepository",
    "SeasonRepository",
    "SpatialLocationRepository",
    "SystemLogRepository",
    "UserLocationRepository",
    "UserPreferenceRepository",
    "UserRepository",
    "UserSessionRepository",
]
