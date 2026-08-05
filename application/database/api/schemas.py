"""Phase 10 API request/response models (pydantic v2)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------- #
# Auth (enterprise)
# ---------------------------------------------------------------------- #
class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


class ResetPasswordRequest(BaseModel):
    email: str


class ResetPasswordConfirmRequest(BaseModel):
    token: str
    new_password: str


class VerifyEmailRequest(BaseModel):
    token: str


class VerifyEmailResponse(BaseModel):
    ok: bool = True
    email: str | None = None


class SessionResponse(BaseModel):
    session_id: str
    ip_address: str | None = None
    user_agent: str | None = None
    created_at: str | None = None
    expires_at: str | None = None
    last_seen_at: str | None = None


# ---------------------------------------------------------------------- #
# Users (preferences + saved locations)
# ---------------------------------------------------------------------- #
class PreferencesUpdate(BaseModel):
    language: str | None = None
    units: str | None = None
    theme: str | None = None
    notification_preferences: dict[str, Any] | None = None
    extra: dict[str, Any] | None = None


class UserLocationCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    lon: float = Field(ge=-180, le=180)
    lat: float = Field(ge=-90, le=90)
    is_default: bool = False
    properties: dict[str, Any] | None = None


# ---------------------------------------------------------------------- #
# Notifications
# ---------------------------------------------------------------------- #
class NotificationSendRequest(BaseModel):
    user_id: int | None = None
    notification_type: str = "generic"
    subject: str | None = None
    body: str | None = None
    channel: str = "email"
    priority: str = "normal"
    metadata: dict[str, Any] | None = None


# ---------------------------------------------------------------------- #
# Feedback
# ---------------------------------------------------------------------- #
class FeedbackSubmit(BaseModel):
    rating: int | None = Field(default=None, ge=1, le=5)
    category: str = "general"
    comment: str | None = None
    issue_type: str | None = None
    prediction_id: int | None = None
    metadata: dict[str, Any] | None = None

    @field_validator("category")
    @classmethod
    def _valid_category(cls, value: str) -> str:
        allowed = {"general", "accuracy", "issue", "feature"}
        if value not in allowed:
            raise ValueError(f"category must be one of {sorted(allowed)}")
        return value


class FeedbackResolve(BaseModel):
    note: str | None = None
    status: str = "resolved"


# ---------------------------------------------------------------------- #
# Registry
# ---------------------------------------------------------------------- #
class ModelVersionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    version: str = Field(min_length=1, max_length=64)
    checkpoint_path: str | None = None
    accuracy: float | None = None
    loss: float | None = None
    git_commit: str | None = None
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    notes: str | None = None
    is_active: bool = False


class DatasetVersionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    version: str = Field(min_length=1, max_length=64)
    source: str | None = None
    description: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    checksum: str | None = None
    is_active: bool = False


# ---------------------------------------------------------------------- #
# Catalog
# ---------------------------------------------------------------------- #
class CropCreate(BaseModel):
    code: str = Field(min_length=1, max_length=32)
    name: str = Field(min_length=1, max_length=255)
    scientific_name: str | None = None
    category: str | None = None
    description: str | None = None
    metadata: dict[str, Any] | None = None


class SeasonCreate(BaseModel):
    code: str = Field(min_length=1, max_length=32)
    name: str = Field(min_length=1, max_length=128)
    start_month: int | None = Field(default=None, ge=1, le=12)
    end_month: int | None = Field(default=None, ge=1, le=12)
    region: str | None = None
    description: str | None = None


# ---------------------------------------------------------------------- #
# Spatial
# ---------------------------------------------------------------------- #
class SpatialLocationCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    lon: float = Field(ge=-180, le=180)
    lat: float = Field(ge=-90, le=90)
    location_type: str = "point"
    properties: dict[str, Any] | None = None
    source: str | None = None


class ResolveRequest(BaseModel):
    lon: float = Field(ge=-180, le=180)
    lat: float = Field(ge=-90, le=90)


# ---------------------------------------------------------------------- #
# Experiments
# ---------------------------------------------------------------------- #
class ExperimentCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    description: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    dataset_version_id: int | None = None
    model_version_id: int | None = None


class ExperimentTransition(BaseModel):
    metrics: dict[str, Any] | None = None
    error: str | None = None


# ---------------------------------------------------------------------- #
# Config store
# ---------------------------------------------------------------------- #
class ConfigSetRequest(BaseModel):
    key: str = Field(min_length=1, max_length=128)
    value: dict[str, Any] | None = None
    category: str | None = None
    description: str | None = None
    is_secret: bool = False
