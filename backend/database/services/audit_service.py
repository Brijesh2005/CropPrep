"""Audit service: record and query the append-only audit trail."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from database.repositories import AuditLogRepository, SystemLogRepository


class AuditService:
    """Append-only audit + system log service."""

    def __init__(self, audit: AuditLogRepository, system: SystemLogRepository) -> None:
        self._audit = audit
        self._system = system

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
    ) -> None:
        await self._audit.record(
            action=action, user_id=user_id, actor_role=actor_role,
            resource_type=resource_type, resource_id=resource_id,
            ip_address=ip_address, user_agent=user_agent, request_id=request_id,
            metadata_=metadata_, outcome=outcome,
        )

    async def query(
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
    ) -> dict[str, Any]:
        rows, total = await self._audit.list_filtered(
            user_id=user_id, action=action, resource_type=resource_type,
            outcome=outcome, date_from=date_from, date_to=date_to,
            limit=limit, offset=offset,
        )
        return {
            "items": [
                {
                    "id": a.id, "action": a.action, "user_id": a.user_id,
                    "actor_role": a.actor_role, "resource_type": a.resource_type,
                    "resource_id": a.resource_id, "ip_address": a.ip_address,
                    "outcome": a.outcome, "metadata": a.metadata_,
                    "created_at": a.created_at.isoformat() if a.created_at else None,
                }
                for a in rows
            ],
            "total": total,
        }

    async def system_log(
        self,
        *,
        level: str,
        message: str,
        logger_name: str | None = None,
        module: str | None = None,
        trace: dict | None = None,
        request_id: str | None = None,
    ) -> None:
        await self._system.record(
            level=level, message=message, logger_name=logger_name,
            module=module, trace=trace, request_id=request_id,
        )
