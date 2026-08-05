"""Notification service: log + fan-out notifications to users."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from database.models.engagement import Notification
from database.repositories import NotificationRepository
from database.services.redis_store import RedisStore


class NotificationService:
    """Create, list and acknowledge notifications."""

    def __init__(self, repository: NotificationRepository, store: RedisStore) -> None:
        self._repo = repository
        self._store = store

    async def send(
        self,
        *,
        user_id: int | None,
        notification_type: str,
        subject: str | None = None,
        body: str | None = None,
        channel: str = "email",
        priority: str = "normal",
        metadata_: dict | None = None,
    ) -> Notification:
        notification = await self._repo.save(
            Notification(
                user_id=user_id,
                channel=channel,
                notification_type=notification_type,
                subject=subject,
                body=body,
                priority=priority,
                metadata_=metadata_ or {},
                status="sent",
                sent_at=datetime.now(timezone.utc),
            )
        )
        await self._invalidate_counts(user_id)
        return notification

    async def list_for_user(
        self, user_id: int, *, limit: int = 50, offset: int = 0, unread_only: bool = False
    ) -> dict[str, Any]:
        rows, total = await self._repo.list_for_user(
            user_id, limit=limit, offset=offset, unread_only=unread_only
        )
        return {
            "items": [
                {
                    "id": n.id,
                    "channel": n.channel,
                    "notification_type": n.notification_type,
                    "subject": n.subject,
                    "body": n.body,
                    "status": n.status,
                    "priority": n.priority,
                    "created_at": n.created_at.isoformat() if n.created_at else None,
                    "read_at": n.read_at.isoformat() if n.read_at else None,
                }
                for n in rows
            ],
            "total": total,
            "unread": await self._repo.count_unread(user_id),
        }

    async def mark_read(self, user_id: int, notification_id: int) -> bool:
        notification = await self._repo.get(notification_id)
        if notification is None or notification.user_id != user_id:
            return False
        await self._repo.mark_read(notification)
        await self._repo.commit()
        await self._invalidate_counts(user_id)
        return True

    async def mark_all_read(self, user_id: int) -> int:
        count = await self._repo.mark_all_read(user_id)
        await self._repo.commit()
        await self._invalidate_counts(user_id)
        return count

    async def _invalidate_counts(self, user_id: int | None) -> None:
        if user_id is not None:
            await self._store.delete(f"notifications:unread:{user_id}")
