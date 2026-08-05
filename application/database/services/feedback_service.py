"""Feedback service: submit, query and resolve user feedback."""

from __future__ import annotations

from typing import Any

from database.models.engagement import Feedback
from database.repositories import FeedbackRepository


class FeedbackService:
    """Manage user feedback on predictions and the platform."""

    def __init__(self, repository: FeedbackRepository) -> None:
        self._repo = repository

    async def submit(
        self,
        *,
        user_id: int | None,
        rating: int | None = None,
        category: str = "general",
        comment: str | None = None,
        issue_type: str | None = None,
        prediction_id: int | None = None,
        metadata_: dict | None = None,
    ) -> Feedback:
        if rating is not None and not (1 <= rating <= 5):
            raise ValueError("rating must be between 1 and 5")
        return await self._repo.save(
            Feedback(
                user_id=user_id,
                rating=rating,
                category=category,
                comment=comment,
                issue_type=issue_type,
                prediction_id=prediction_id,
                metadata_=metadata_ or {},
                status="open",
            )
        )

    async def list(
        self,
        *,
        status: str | None = None,
        category: str | None = None,
        user_id: int | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        rows, total = await self._repo.list_filtered(
            status=status, category=category, user_id=user_id, limit=limit, offset=offset
        )
        return {
            "items": [
                {
                    "id": f.id, "user_id": f.user_id, "rating": f.rating,
                    "category": f.category, "comment": f.comment,
                    "issue_type": f.issue_type, "status": f.status,
                    "prediction_id": f.prediction_id,
                    "metadata": f.metadata_,
                    "resolved_by": f.resolved_by, "resolved_at": f.resolved_at,
                    "resolution_note": f.resolution_note,
                    "created_at": f.created_at.isoformat() if f.created_at else None,
                }
                for f in rows
            ],
            "total": total,
        }

    async def resolve(self, feedback_id: int, *, resolved_by: int, note: str | None = None) -> Feedback | None:
        feedback = await self._repo.get(feedback_id)
        if feedback is None:
            return None
        await self._repo.resolve(feedback, resolved_by=resolved_by, note=note)
        await self._repo.commit()
        return feedback
