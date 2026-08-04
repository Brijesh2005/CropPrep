"""Notification routes: inbox, unread counts, acknowledgements."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query

from app.core.security import ROLE_ADMIN
from app.dependencies.enterprise import get_notification_service
from app.dependencies.security import get_current_user, require_role
from database.api.schemas import NotificationSendRequest
from database.services.notification_service import NotificationService

router = APIRouter(prefix="/notifications", tags=["notifications"])


@router.get(
    "",
    summary="List notifications",
    description="The authenticated user's notification inbox.",
)
async def list_notifications(
    unread_only: bool = Query(default=False),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    user: Any = Depends(get_current_user),
    service: NotificationService = Depends(get_notification_service),
) -> dict[str, Any]:
    return await service.list_for_user(
        user.id, limit=limit, offset=offset, unread_only=unread_only
    )


@router.get(
    "/unread-count",
    summary="Unread notification count",
)
async def unread_count(
    user: Any = Depends(get_current_user),
    service: NotificationService = Depends(get_notification_service),
) -> dict[str, int]:
    count = await service.list_for_user(user.id, limit=1)
    return {"unread": count["unread"]}


@router.post(
    "/{notification_id}/read",
    summary="Mark a notification as read",
)
async def mark_read(
    notification_id: int,
    user: Any = Depends(get_current_user),
    service: NotificationService = Depends(get_notification_service),
) -> dict[str, Any]:
    ok = await service.mark_read(user.id, notification_id)
    return {"ok": ok}


@router.post(
    "/read-all",
    summary="Mark all notifications as read",
)
async def mark_all_read(
    user: Any = Depends(get_current_user),
    service: NotificationService = Depends(get_notification_service),
) -> dict[str, int]:
    count = await service.mark_all_read(user.id)
    return {"marked": count}


@router.post(
    "",
    summary="Send a notification (admin)",
    description="Creates a notification log entry for a user (admin only).",
)
async def send_notification(
    body: NotificationSendRequest,
    _: Any = Depends(require_role(ROLE_ADMIN)),
    service: NotificationService = Depends(get_notification_service),
) -> dict[str, Any]:
    notification = await service.send(
        user_id=body.user_id,
        notification_type=body.notification_type,
        subject=body.subject,
        body=body.body,
        channel=body.channel,
        priority=body.priority,
        metadata_=body.metadata,
    )
    return {"id": notification.id}
