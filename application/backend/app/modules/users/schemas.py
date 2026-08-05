"""Users module schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class UserResponse(BaseModel):
    id: int
    email: str
    full_name: str | None = None
    role: str
    is_active: bool = True


class ProfileUpdateRequest(BaseModel):
    full_name: str | None = Field(default=None, max_length=255)
    password: str | None = Field(default=None, min_length=8, max_length=128)
