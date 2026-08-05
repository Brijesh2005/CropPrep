"""Engagement repositories: notifications and feedback."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import func, select

from database.models.engagement import Feedback, Notification
from database.repositories.base import DataRepository


class NotificationRepository(DataRepository[Notification]):
    model = Notification

    async def list_for_user(
        self, user_id: int, *, limit: int = 50, offset: int = 0, unread_only: bool = False
    ) -> tuple[list[Notification], int]:
        stmt = select(Notification).where(Notification.user_id == user_id)
        if unread_only:
            stmt = stmt.where(Notification.status != "read")
        stmt = stmt.order_by(Notification.created_at.desc())
        return await self.paginate(stmt, limit=limit, offset=offset)

    async def count_unread(self, user_id: int) -> int:
        return await self.count_rows(
            Notification.user_id == user_id,
            Notification.status != "read",
        )

    async def mark_read(self, notification: Notification) -> None:
        notification.status = "read"
        notification.read_at = datetime.now(timezone.utc)
        await self.session.flush()

    async def mark_all_read(self, user_id: int) -> int:
        result = await self.session.execute(
            select(Notification).where(Notification.user_id == user_id, Notification.status != "read")
        )
        rows = list(result.scalars().all())
        now = datetime.now(timezone.utc)
        for row in rows:
            row.status = "read"
            row.read_at = now
        await self.session.flush()
        return len(rows)


class FeedbackRepository(DataRepository[Feedback]):
    model = Feedback

    async def list_filtered(
        self,
        *,
        status: str | None = None,
        category: str | None = None,
        user_id: int | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Feedback], int]:
        stmt = select(Feedback)
        if status:
            stmt = stmt.where(Feedback.status == status)
        if category:
            stmt = stmt.where(Feedback.category == category)
        if user_id is not None:
            stmt = stmt.where(Feedback.user_id == user_id)
        stmt = stmt.order_by(Feedback.created_at.desc())
        return await self.paginate(stmt, limit=limit, offset=offset)

    async def average_rating(self) -> float | None:
        result = await self.session.execute(select(func.avg(Feedback.rating)))
        value = result.scalar_one_or_none()
        return round(float(value), 2) if value is not None else None

    async def resolve(
        self, feedback: Feedback, *, resolved_by: int, note: str | None = None
    ) -> Feedback:
        feedback.status = "resolved"
        feedback.resolved_by = resolved_by
        feedback.resolved_at = datetime.now(timezone.utc)
        if note:
            feedback.resolution_note = note
        await self.session.flush()
        return feedback
