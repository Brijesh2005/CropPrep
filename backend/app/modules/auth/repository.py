"""Auth module repository (re-exports the shared user repository)."""

from __future__ import annotations

from app.repositories.user import UserRepository

__all__ = ["UserRepository"]
