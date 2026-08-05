"""Enterprise authentication service (Phase 10).

Drop-in replacement for the Phase 8 :class:`app.modules.auth.service.AuthService`
(keeps ``register`` / ``login`` / ``refresh`` / ``logout`` signatures) and adds
the full enterprise lifecycle:

* Argon2id password hashing (legacy passlib hashes still verify),
* account lockout after repeated failures,
* persisted, rotating refresh tokens with reuse detection,
* user sessions (DB + Redis) with a per-user session cap,
* password reset / email verification (single-use, hashed tokens),
* audit logging of security events.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import PasswordPolicySettings, SecuritySettings
from app.core.exceptions import AuthenticationError, TokenError
from app.core.security import decode_token
from app.models.user import User
from app.modules.auth.exceptions import EmailAlreadyRegisteredError
from database.models.tokens import EmailVerificationToken, PasswordResetToken, RefreshToken
from database.repositories import (
    AuditLogRepository,
    EmailVerificationTokenRepository,
    PasswordResetTokenRepository,
    RefreshTokenRepository,
    UserRepository,
    UserSessionRepository,
)
from database.security import (
    AccountLockout,
    PasswordPolicy,
    PasswordService,
    SessionService,
    TokenService,
)
from database.services.redis_store import RedisStore

AUDIT_ACTIONS = {
    "login": "user.login",
    "login_failed": "user.login.failed",
    "logout": "user.logout",
    "refresh": "user.refresh",
    "register": "user.register",
    "reset_password": "user.password.reset",
    "change_password": "user.password.change",
    "verify_email": "user.email.verify",
    "revoke_session": "user.session.revoke",
}


class AuthService:
    """Enterprise auth lifecycle (register, login, refresh, sessions, recovery)."""

    def __init__(
        self,
        session: AsyncSession,
        security_settings: SecuritySettings,
        password_settings: PasswordPolicySettings,
        redis_store: RedisStore,
    ) -> None:
        self._session = session
        self._settings = security_settings
        self._users = UserRepository(session)
        self._refresh_tokens = RefreshTokenRepository(session)
        self._reset_tokens = PasswordResetTokenRepository(session)
        self._verification_tokens = EmailVerificationTokenRepository(session)
        self._sessions = UserSessionRepository(session)
        self._audit = AuditLogRepository(session)
        self._token_service = TokenService(security_settings)
        self._passwords = PasswordService(
            time_cost=security_settings.argon2_time_cost,
            memory_kib=security_settings.argon2_memory_kib,
            parallelism=security_settings.argon2_parallelism,
            legacy_scheme=security_settings.password_scheme,
        )
        self._policy = PasswordPolicy(password_settings)
        self._lockout = AccountLockout(
            password_settings.max_failed_attempts, password_settings.lockout_minutes
        )
        self._session_service = SessionService(self._sessions, redis_store, security_settings)

    # ------------------------------------------------------------------ #
    # Registration / login / refresh / logout
    # ------------------------------------------------------------------ #
    async def register(
        self, *, email: str, password: str, full_name: str | None = None, **_: Any
    ) -> User:
        email = email.strip().lower()
        self._policy.validate(password, email=email)
        if await self._users.exists(email):
            raise EmailAlreadyRegisteredError("email is already registered", detail=email)
        user = User(
            email=email,
            hashed_password=self._passwords.hash(password),
            full_name=full_name,
            role="user",
            is_active=True,
            is_email_verified=False,
        )
        await self._users.add(user)
        await self._users.commit()
        await self._users.refresh(user)
        await self._audit.record(
            action=AUDIT_ACTIONS["register"], user_id=user.id, actor_role=user.role
        )
        await self._session.commit()
        return user

    async def login(
        self,
        *,
        email: str,
        password: str,
        ip: str | None = None,
        user_agent: str | None = None,
    ) -> dict[str, Any]:
        user = await self._users.get_by_email(email)
        if user is None:
            raise AuthenticationError("invalid email or password")
        if self._lockout.is_locked(user.failed_login_attempts, user.locked_until):
            raise AuthenticationError(
                "account temporarily locked; try again later",
                detail={
                    "retry_after_seconds": self._lockout.lockout_expires_in_seconds(
                        user.locked_until
                    )
                    if user.locked_until
                    else None,
                },
            )
        if not user.is_active:
            raise AuthenticationError("user account is disabled")
        if not self._passwords.verify(password, user.hashed_password):
            attempts = await self._users.record_login_failure(user)
            locked_until = self._lockout.next_lockout(attempts)
            if locked_until is not None:
                await self._users.lock_account(user, locked_until)
            await self._audit.record(
                action=AUDIT_ACTIONS["login_failed"], user_id=user.id,
                actor_role=user.role, ip_address=ip, outcome="failure",
            )
            await self._session.commit()
            raise AuthenticationError("invalid email or password")
        await self._users.record_login_success(user)
        return await self._issue_tokens(user, ip=ip, user_agent=user_agent, action="login")

    async def refresh(
        self, refresh_token: str, *, ip: str | None = None, user_agent: str | None = None
    ) -> dict[str, Any]:
        token_hash = TokenService.hash_token(refresh_token)
        stored = await self._refresh_tokens.get_by_hash(token_hash)
        if stored is None:
            raise TokenError("refresh token is invalid")
        if stored.revoked_at is not None:
            # Token reuse: a previously-rotated token was replayed.
            await self._refresh_tokens.revoke_all_for_user(stored.user_id)
            await self._sessions.revoke_all(stored.user_id)
            await self._session.commit()
            raise TokenError("refresh token has been revoked")
        try:
            payload = decode_token(refresh_token, self._settings, expected="refresh")
        except TokenError:
            await self._refresh_tokens.revoke(stored)
            await self._session.commit()
            raise
        user = await self._users.get(int(payload["sub"]))
        if user is None or not user.is_active:
            raise TokenError("user no longer active")
        if int(payload["sub"]) != stored.user_id:
            raise TokenError("refresh token does not match the session")
        return await self._issue_tokens(
            user, ip=ip, user_agent=user_agent, action="refresh", previous_token=stored
        )

    async def logout(
        self, access_token: str, *, refresh_token: str | None = None,
        ip: str | None = None,
    ) -> None:
        if refresh_token:
            stored = await self._refresh_tokens.get_by_hash(
                TokenService.hash_token(refresh_token)
            )
            if stored is not None:
                user = await self._users.get(stored.user_id)
                await self._refresh_tokens.revoke(stored)
                if stored.id is not None:
                    session = await self._sessions.get_by_refresh_token_id(stored.id)
                    if session is not None:
                        await self._sessions.revoke(session)
                if user is not None:
                    await self._audit.record(
                        action=AUDIT_ACTIONS["logout"], user_id=user.id,
                        actor_role=user.role, ip_address=ip,
                    )
                await self._session.commit()

    # ------------------------------------------------------------------ #
    # Password recovery
    # ------------------------------------------------------------------ #
    async def request_password_reset(self, *, email: str, ip: str | None = None) -> dict[str, Any]:
        user = await self._users.get_by_email(email)
        if user is None:
            return {"ok": True, "sent": False}
        raw, token_hash = self._token_service.new_opaque_token()
        await self._reset_tokens.add(
            PasswordResetToken(
                user_id=user.id,
                token_hash=token_hash,
                expires_at=datetime.now(timezone.utc)
                + timedelta(minutes=self._settings.password_policy.reset_token_ttl_minutes),
            )
        )
        await self._audit.record(
            action=AUDIT_ACTIONS["reset_password"], user_id=user.id,
            actor_role=user.role, ip_address=ip,
        )
        await self._session.commit()
        return {"ok": True, "sent": True, "token": raw}

    async def reset_password(
        self, *, token: str, new_password: str, ip: str | None = None
    ) -> None:
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        stored = await self._reset_tokens.get_active_by_hash(token_hash)
        if stored is None:
            raise TokenError("reset token is invalid or has expired")
        user = await self._users.get(stored.user_id)
        if user is None or not user.is_active:
            raise TokenError("user no longer active")
        self._policy.validate(new_password, email=user.email)
        await self._users.set_password(user, self._passwords.hash(new_password))
        await self._reset_tokens.consume(stored, ip=ip)
        await self._refresh_tokens.revoke_all_for_user(user.id)
        await self._sessions.revoke_all(user.id)
        await self._audit.record(
            action=AUDIT_ACTIONS["reset_password"], user_id=user.id,
            actor_role=user.role, ip_address=ip,
        )
        await self._session.commit()

    async def change_password(
        self, *, user: User, current_password: str, new_password: str, ip: str | None = None
    ) -> None:
        if not self._passwords.verify(current_password, user.hashed_password):
            raise AuthenticationError("current password is incorrect")
        self._policy.validate(new_password, email=user.email)
        await self._users.set_password(user, self._passwords.hash(new_password))
        await self._refresh_tokens.revoke_all_for_user(user.id)
        await self._sessions.revoke_all(user.id)
        await self._audit.record(
            action=AUDIT_ACTIONS["change_password"], user_id=user.id,
            actor_role=user.role, ip_address=ip,
        )
        await self._session.commit()

    # ------------------------------------------------------------------ #
    # Email verification / sessions
    # ------------------------------------------------------------------ #
    async def request_email_verification(self, *, user: User) -> dict[str, Any]:
        raw, token_hash = self._token_service.new_opaque_token()
        await self._verification_tokens.add(
            EmailVerificationToken(
                user_id=user.id,
                token_hash=token_hash,
                expires_at=datetime.now(timezone.utc)
                + timedelta(hours=self._settings.password_policy.email_verify_ttl_hours),
            )
        )
        await self._session.commit()
        return {"ok": True, "token": raw}

    async def verify_email(self, *, token: str, ip: str | None = None) -> User:
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        stored = await self._verification_tokens.get_active_by_hash(token_hash)
        if stored is None:
            raise TokenError("verification token is invalid or has expired")
        user = await self._users.get(stored.user_id)
        if user is None:
            raise TokenError("user no longer exists")
        await self._users.mark_email_verified(user)
        await self._verification_tokens.consume(stored)
        await self._audit.record(
            action=AUDIT_ACTIONS["verify_email"], user_id=user.id,
            actor_role=user.role, ip_address=ip,
        )
        await self._session.commit()
        return user

    async def list_sessions(self, user_id: int) -> list[dict[str, Any]]:
        sessions = await self._sessions.list_active(user_id)
        return [
            {
                "session_id": s.session_id,
                "ip_address": s.ip_address,
                "user_agent": s.user_agent,
                "created_at": s.created_at.isoformat() if s.created_at else None,
                "expires_at": s.expires_at.isoformat() if s.expires_at else None,
                "last_seen_at": s.last_seen_at.isoformat() if s.last_seen_at else None,
            }
            for s in sessions
        ]

    async def revoke_session(self, user_id: int, session_id: str) -> bool:
        session = await self._sessions.get_by_session_id(session_id)
        if session is None or session.user_id != user_id:
            return False
        await self._sessions.revoke(session)
        await self._audit.record(
            action=AUDIT_ACTIONS["revoke_session"], user_id=user_id,
            resource_type="user_session", resource_id=session_id,
        )
        await self._session.commit()
        return True

    async def revoke_all_sessions(self, user_id: int, *, except_session_id: str | None = None) -> int:
        count = await self._sessions.revoke_all(user_id, except_session_id=except_session_id)
        await self._audit.record(
            action=AUDIT_ACTIONS["revoke_session"], user_id=user_id,
            resource_type="user_session", metadata_={"revoked": count},
        )
        await self._session.commit()
        return count

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    async def _issue_tokens(
        self,
        user: User,
        *,
        ip: str | None = None,
        user_agent: str | None = None,
        action: str,
        previous_token: RefreshToken | None = None,
    ) -> dict[str, Any]:
        await self._session_service.enforce_max_sessions(user.id)
        access = self._token_service.create_access(user.id, user.role)
        raw, jti, token_hash = self._token_service.create_refresh(user.id, user.role)
        if previous_token is not None:
            await self._refresh_tokens.revoke(previous_token, replaced_by_jti=jti)
        stored_token = await self._refresh_tokens.add(
            RefreshToken(
                user_id=user.id, jti=jti, token_hash=token_hash,
                expires_at=datetime.now(timezone.utc) + timedelta(
                    days=self._settings.refresh_token_expire_days
                ),
                ip_address=ip, user_agent=user_agent,
            )
        )
        session = await self._session_service.create(
            user, refresh_token_id=stored_token.id, ip=ip, user_agent=user_agent
        )
        await self._audit.record(
            action=AUDIT_ACTIONS.get(action, action), user_id=user.id,
            actor_role=user.role, ip_address=ip,
        )
        await self._session.commit()
        await self._users.refresh(user)
        return {
            "access_token": access,
            "refresh_token": raw,
            "token_type": "bearer",
            "expires_in": self._settings.access_token_expire_minutes * 60,
            "session_id": session.session_id,
            "user": user.profile_dict(),
        }
