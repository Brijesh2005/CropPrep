"""Enterprise data-layer models (Phase 10).

Importing this package registers every enterprise table on the shared
:class:`app.core.database.Base` metadata. Core Phase 8 models (``users``,
``predictions``, ``explanations``) are extended in ``app.models``.
"""

from __future__ import annotations

from database.models.access import Permission, Role, role_permissions
from database.models.catalog import Crop, Season
from database.models.configuration import AppConfiguration
from database.models.engagement import Feedback, Notification
from database.models.experiments import ResearchExperiment
from database.models.logging_models import AuditLog, SystemLog
from database.models.metadata import PredictionMetadata
from database.models.profile import UserLocation, UserPreference
from database.models.registry import DatasetVersion, ModelVersion
from database.models.session import UserSession
from database.models.spatial import AdministrativeBoundary, SpatialLocation
from database.models.tokens import (
    EmailVerificationToken,
    PasswordResetToken,
    RefreshToken,
)

__all__ = [
    "Role",
    "Permission",
    "role_permissions",
    "Crop",
    "Season",
    "AppConfiguration",
    "Feedback",
    "Notification",
    "ResearchExperiment",
    "AuditLog",
    "SystemLog",
    "PredictionMetadata",
    "UserLocation",
    "UserPreference",
    "DatasetVersion",
    "ModelVersion",
    "UserSession",
    "AdministrativeBoundary",
    "SpatialLocation",
    "EmailVerificationToken",
    "PasswordResetToken",
    "RefreshToken",
]
