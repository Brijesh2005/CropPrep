"""Feedback routes: submit, query and resolve feedback."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query

from app.core.security import ROLE_ADMIN
from app.dependencies.enterprise import get_feedback_service
from app.dependencies.security import get_current_user, get_current_user_optional, require_role
from database.api.schemas import FeedbackResolve, FeedbackSubmit
from database.services.feedback_service import FeedbackService

router = APIRouter(prefix="/feedback", tags=["feedback"])


@router.post(
    "",
    summary="Submit feedback",
    description="Submit feedback (optionally tied to a prediction).",
)
async def submit_feedback(
    body: FeedbackSubmit,
    user: Any = Depends(get_current_user_optional),
    service: FeedbackService = Depends(get_feedback_service),
) -> dict[str, Any]:
    feedback = await service.submit(
        user_id=user.id if user else None,
        rating=body.rating,
        category=body.category,
        comment=body.comment,
        issue_type=body.issue_type,
        prediction_id=body.prediction_id,
        metadata_=body.metadata,
    )
    return {"id": feedback.id, "status": feedback.status}


@router.get(
    "",
    summary="List feedback (admin)",
    description="Filtered feedback queue for moderation (admin only).",
)
async def list_feedback(
    status: str | None = Query(default=None),
    category: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    _: Any = Depends(require_role(ROLE_ADMIN)),
    service: FeedbackService = Depends(get_feedback_service),
) -> dict[str, Any]:
    return await service.list(status=status, category=category, limit=limit, offset=offset)


@router.post(
    "/{feedback_id}/resolve",
    summary="Resolve feedback (admin)",
)
async def resolve_feedback(
    body: FeedbackResolve,
    feedback_id: int,
    user: Any = Depends(require_role(ROLE_ADMIN)),
    service: FeedbackService = Depends(get_feedback_service),
) -> dict[str, Any]:
    resolved = await service.resolve(feedback_id, resolved_by=user.id, note=body.note)
    if resolved is None:
        return {"ok": False}
    return {"ok": True, "status": body.status}
