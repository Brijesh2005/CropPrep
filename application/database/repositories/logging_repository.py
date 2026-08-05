"""Audit and system log repositories."""

from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import func, select

from database.models.logging_models import AuditLog, SystemLog
from database.repositories.base import DataRepository


class AuditLogRepository(DataRepository[AuditLog]):
    model = AuditLog

    async def record(
        self,
        *,
        action: str,
        user_id: int | None = None,
        actor_role: str | None = None,
        resource_type: str | None = None,
        resource_id: str | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
        request_id: str | None = None,
        metadata_: dict | None = None,
        outcome: str = "success",
    ) -> AuditLog:
        return await self.add(
            AuditLog(
                action=action,
                user_id=user_id,
                actor_role=actor_role,
                resource_type=resource_type,
                resource_id=resource_id,
                ip_address=ip_address,
                user_agent=user_agent,
                request_id=request_id,
                metadata_=metadata_ or {},
                outcome=outcome,
            )
        )

    async def list_filtered(
        self,
        *,
        user_id: int | None = None,
        action: str | None = None,
        resource_type: str | None = None,
        outcome: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[AuditLog], int]:
        stmt = select(AuditLog)
        if user_id is not None:
            stmt = stmt.where(AuditLog.user_id == user_id)
        if action:
            stmt = stmt.where(AuditLog.action == action)
        if resource_type:
            stmt = stmt.where(AuditLog.resource_type == resource_type)
        if outcome:
            stmt = stmt.where(AuditLog.outcome == outcome)
        if date_from:
            stmt = stmt.where(AuditLog.created_at >= date_from)
        if date_to:
            stmt = stmt.where(AuditLog.created_at <= date_to)
        stmt = stmt.order_by(AuditLog.created_at.desc())
        return await self.paginate(stmt, limit=limit, offset=offset)

    async def recent_actions(self, *, hours: int = 24) -> list[tuple[str, int]]:
        since = datetime.utcnow() - timedelta(hours=hours)
        result = await self.session.execute(
            select(AuditLog.action, func.count(AuditLog.id))
            .where(AuditLog.created_at >= since)
            .group_by(AuditLog.action)
            .order_by(func.count(AuditLog.id).desc())
        )
        return [(action, int(count)) for action, count in result.all()]


class SystemLogRepository(DataRepository[SystemLog]):
    model = SystemLog

    async def record(
        self,
        *,
        level: str,
        message: str,
        logger_name: str | None = None,
        module: str | None = None,
        trace: dict | None = None,
        request_id: str | None = None,
    ) -> SystemLog:
        return await self.add(
            SystemLog(
                level=level,
                logger_name=logger_name,
                module=module,
                message=message,
                trace=trace,
                request_id=request_id,
            )
        )

    async def list_filtered(
        self,
        *,
        level: str | None = None,
        module: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[SystemLog], int]:
        stmt = select(SystemLog)
        if level:
            stmt = stmt.where(SystemLog.level == level)
        if module:
            stmt = stmt.where(SystemLog.module == module)
        stmt = stmt.order_by(SystemLog.created_at.desc())
        return await self.paginate(stmt, limit=limit, offset=offset)

    async def purge_older_than(self, days: int) -> int:
        cutoff = datetime.utcnow() - timedelta(days=days)
        result = await self.session.execute(
            SystemLog.__table__.delete().where(SystemLog.created_at < cutoff)
        )
        await self.session.flush()
        return result.rowcount or 0
