"""JWT token service (Phase 10).

Wraps the Phase 8 ``core.security`` JWT helpers and adds:
* ``jti`` claims for revocation / rotation,
* refresh-token hashing (SHA-256) so tokens are never stored in plain text.
"""

from __future__ import annotations

import hashlib
import secrets
import uuid
from typing import Any

from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
)
from app.core.config import SecuritySettings


class TokenService:
    """Issues access/refresh JWTs and hashes refresh tokens."""

    def __init__(self, settings: SecuritySettings) -> None:
        self._settings = settings

    def create_access(self, user_id: int, role: str, *, extra: dict[str, Any] | None = None) -> str:
        jti = uuid.uuid4().hex
        return create_access_token(
            str(user_id), role=role, settings=self._settings, extra={"jti": jti, **(extra or {})}
        )

    def create_refresh(self, user_id: int, role: str) -> tuple[str, str, str]:
        """Return ``(raw_token, jti, token_hash)``.

        The ``jti`` is embedded in the token payload so two tokens issued within
        the same second are still unique (and their SHA-256 hashes never collide
        on the ``token_hash`` unique column).
        """
        jti = uuid.uuid4().hex
        raw = create_refresh_token(
            str(user_id), role=role, settings=self._settings, extra={"jti": jti}
        )
        return raw, jti, TokenService.hash_token(raw)

    def decode(self, token: str, *, expected: str | None = None) -> dict[str, Any]:
        return decode_token(token, self._settings, expected=expected)

    @staticmethod
    def hash_token(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    @staticmethod
    def new_opaque_token() -> tuple[str, str]:
        """Return ``(raw_token, sha256_hash)`` for single-use verification tokens."""
        raw = secrets.token_urlsafe(48)
        return raw, hashlib.sha256(raw.encode("utf-8")).hexdigest()
