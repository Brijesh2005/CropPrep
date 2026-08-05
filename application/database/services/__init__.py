"""Service layer (Phase 10).

Note: :class:`database.services.auth.AuthService` is intentionally not re-exported
here because it depends on ``database.security``, which itself imports
``database.services.redis_store`` (import-cycle avoidance). Import it directly
from ``database.services.auth``.
"""

from __future__ import annotations

from database.services.analytics import AnalyticsService
from database.services.audit_service import AuditService
from database.services.catalog_service import CatalogService
from database.services.config_service import ConfigService
from database.services.experiment_service import ExperimentService
from database.services.feedback_service import FeedbackService
from database.services.geo import bbox_from_center, haversine_km, validate_coordinates
from database.services.notification_service import NotificationService
from database.services.prediction_service import PredictionHistoryService
from database.services.profile import ProfileService
from database.services.redis_store import MemoryStore, RedisStore, build_redis_store
from database.services.registry_service import RegistryService
from database.services.spatial_service import SpatialService

__all__ = [
    "AnalyticsService",
    "AuditService",
    "CatalogService",
    "ConfigService",
    "ExperimentService",
    "FeedbackService",
    "MemoryStore",
    "NotificationService",
    "PredictionHistoryService",
    "ProfileService",
    "RedisStore",
    "RegistryService",
    "SpatialService",
    "bbox_from_center",
    "build_redis_store",
    "haversine_km",
    "validate_coordinates",
]
