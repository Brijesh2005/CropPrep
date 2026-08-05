"""Security utilities: JWT, password hashing, RBAC roles.

* JWT access + refresh tokens (OAuth2 bearer / password flow) via ``jose``.
* Password hashing / verification via ``passlib`` (PBKDF2-SHA256 by default,
  no optional bcrypt dependency).
* Role constants for RBAC.
"""

from __future__ import annotations

import datetime
from typing import Any

from jose import JWTError, jwt
from passlib.context import CryptContext

from .config import SecuritySettings
from .exceptions import AuthenticationError, TokenError

#: RBAC roles (ordered by privilege).
ROLE_ADMIN = "admin"
ROLE_ANALYST = "analyst"
ROLE_USER = "user"
#: Phase 10 enterprise roles.
ROLE_SUPER_ADMIN = "super_admin"
ROLE_DATASET_MANAGER = "dataset_manager"

ALL_ROLES = (ROLE_USER, ROLE_ANALYST, ROLE_DATASET_MANAGER, ROLE_ADMIN, ROLE_SUPER_ADMIN)


def _pwd_context(scheme: str) -> CryptContext:
    return CryptContext(schemes=[scheme], deprecated="auto")


def hash_password(password: str, scheme: str = "pbkdf2_sha256") -> str:
    """Hash a password (never store plain text)."""
    return _pwd_context(scheme).hash(password)


def verify_password(password: str, hashed: str, scheme: str = "pbkdf2_sha256") -> bool:
    """Verify a password against its stored hash.

    Handles both the Phase 8 passlib hashes and the Phase 10 Argon2id hashes
    (produced by password change / reset / seeded accounts).
    """
    if hashed.startswith("$argon2"):
        from database.security.passwords import PasswordService

        try:
            return PasswordService().verify(password, hashed)
        except Exception:  # pragma: no cover - defensive
            return False
    return _pwd_context(scheme).verify(password, hashed)


def create_access_token(
    subject: str,
    *,
    role: str,
    settings: SecuritySettings,
    extra: dict[str, Any] | None = None,
) -> str:
    """Create a short-lived access token."""
    expire = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(
        minutes=settings.access_token_expire_minutes
    )
    return _encode(
        {"sub": subject, "role": role, "type": "access", **(extra or {})},
        expire,
        settings,
    )


def create_refresh_token(
    subject: str,
    *,
    role: str,
    settings: SecuritySettings,
    extra: dict[str, Any] | None = None,
) -> str:
    """Create a long-lived refresh token."""
    expire = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(
        days=settings.refresh_token_expire_days
    )
    return _encode(
        {"sub": subject, "role": role, "type": "refresh", **(extra or {})}, expire, settings
    )


def _encode(payload: dict[str, Any], expire: datetime.datetime, settings: SecuritySettings) -> str:
    data = dict(payload)
    data.update(
        {
            "exp": expire,
            "iat": datetime.datetime.now(datetime.timezone.utc),
            "iss": settings.issuer,
        }
    )
    return jwt.encode(data, settings.secret_key, algorithm=settings.algorithm)


def decode_token(token: str, settings: SecuritySettings, *, expected: str | None = None) -> dict[str, Any]:
    """Decode and validate a JWT (optionally requiring a token type)."""
    try:
        payload = jwt.decode(
            token, settings.secret_key, algorithms=[settings.algorithm],
            issuer=settings.issuer,
        )
    except JWTError as exc:
        raise TokenError("invalid or expired token", detail=str(exc)) from exc
    if expected is not None and payload.get("type") != expected:
        raise TokenError(f"expected a {expected} token")
    if not payload.get("sub"):
        raise TokenError("token missing subject")
    return payload


def get_public_key_fingerprint(settings: SecuritySettings) -> str:
    """A short fingerprint for the token issuer (used by /health)."""
    import hashlib

    return hashlib.sha256(settings.secret_key.encode()).hexdigest()[:12]


class RBAC:
    """Role-based access helpers."""

    @staticmethod
    def can(required_role: str, user_role: str | None) -> bool:
        """Whether ``user_role`` satisfies ``required_role``."""
        if user_role is None:
            return False
        order = {
            ROLE_USER: 0,
            ROLE_ANALYST: 1,
            ROLE_DATASET_MANAGER: 2,
            ROLE_ADMIN: 3,
            ROLE_SUPER_ADMIN: 4,
        }
        return order.get(user_role, -1) >= order.get(required_role, 99)


def authenticate_password(email: str, password: str, hashed: str, scheme: str = "pbkdf2_sha256") -> None:
    """Raise :class:`AuthenticationError` unless the password matches."""
    if not verify_password(password, hashed, scheme):
        raise AuthenticationError("invalid email or password")
