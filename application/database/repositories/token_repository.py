"""Token repositories: refresh tokens, email verification, password reset."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import delete, select

from database.models.tokens import (
    EmailVerificationToken,
    PasswordResetToken,
    RefreshToken,
)
from database.repositories.base import DataRepository


class RefreshTokenRepository(DataRepository[RefreshToken]):
    model = RefreshToken

    async def get_by_jti(self, jti: str) -> RefreshToken | None:
        result = await self.session.execute(select(RefreshToken).where(RefreshToken.jti == jti))
        return result.scalar_one_or_none()

    async def get_by_hash(self, token_hash: str) -> RefreshToken | None:
        result = await self.session.execute(
            select(RefreshToken).where(RefreshToken.token_hash == token_hash)
        )
        return result.scalar_one_or_none()

    async def revoke(self, token: RefreshToken, *, replaced_by_jti: str | None = None) -> None:
        token.revoked_at = datetime.utcnow()
        token.replaced_by_jti = replaced_by_jti
        await self.session.flush()

    async def revoke_all_for_user(self, user_id: int, *, except_jti: str | None = None) -> None:
        stmt = select(RefreshToken.id).where(
            RefreshToken.user_id == user_id, RefreshToken.revoked_at.is_(None)
        )
        if except_jti:
            stmt = stmt.where(RefreshToken.jti != except_jti)
        await self.session.execute(
            delete(RefreshToken).where(RefreshToken.id.in_(stmt))
        )
        await self.session.flush()

    async def purge_expired(self, now: datetime | None = None) -> int:
        now = now or datetime.utcnow()
        result = await self.session.execute(
            delete(RefreshToken).where(RefreshToken.expires_at < now)
        )
        await self.session.flush()
        return result.rowcount or 0


class EmailVerificationTokenRepository(DataRepository[EmailVerificationToken]):
    model = EmailVerificationToken

    async def get_active_by_hash(self, token_hash: str) -> EmailVerificationToken | None:
        result = await self.session.execute(
            select(EmailVerificationToken).where(
                EmailVerificationToken.token_hash == token_hash,
                EmailVerificationToken.consumed_at.is_(None),
                EmailVerificationToken.expires_at > datetime.utcnow(),
            )
        )
        return result.scalar_one_or_none()

    async def consume(self, token: EmailVerificationToken) -> None:
        token.consumed_at = datetime.utcnow()
        await self.session.flush()


class PasswordResetTokenRepository(DataRepository[PasswordResetToken]):
    model = PasswordResetToken

    async def get_active_by_hash(self, token_hash: str) -> PasswordResetToken | None:
        result = await self.session.execute(
            select(PasswordResetToken).where(
                PasswordResetToken.token_hash == token_hash,
                PasswordResetToken.consumed_at.is_(None),
                PasswordResetToken.expires_at > datetime.utcnow(),
            )
        )
        return result.scalar_one_or_none()

    async def consume(self, token: PasswordResetToken, *, ip: str | None = None) -> None:
        token.consumed_at = datetime.utcnow()
        token.used_from_ip = ip
        await self.session.flush()
