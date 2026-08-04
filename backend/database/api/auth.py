"""Enterprise auth routes: password recovery, email verification, sessions.

These complement the Phase 8 ``/auth`` routes (register / login / refresh /
logout) without duplicating them.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request

from app.dependencies.enterprise import get_enterprise_auth_service
from app.dependencies.security import get_current_user
from app.models.user import User
from database.api.schemas import (
    ChangePasswordRequest,
    ResetPasswordConfirmRequest,
    ResetPasswordRequest,
    SessionResponse,
    VerifyEmailRequest,
    VerifyEmailResponse,
)
from database.services.auth import AuthService

router = APIRouter(prefix="/auth", tags=["authentication-enterprise"])


def _client(request: Request) -> dict[str, str | None]:
    return {
        "ip": request.client.host if request.client else None,
        "user_agent": request.headers.get("user-agent"),
    }


@router.post(
    "/password/change",
    summary="Change the current password",
    description="Verifies the current password, applies the password policy and "
    "revokes all existing refresh tokens and sessions.",
)
async def change_password(
    body: ChangePasswordRequest,
    request: Request,
    user: Any = Depends(get_current_user),
    service: AuthService = Depends(get_enterprise_auth_service),
) -> dict[str, str]:
    await service.change_password(
        user=user,
        current_password=body.current_password,
        new_password=body.new_password,
        ip=_client(request)["ip"],
    )
    return {"message": "password changed; other sessions have been revoked"}


@router.post(
    "/password/reset",
    summary="Request a password reset",
    description="Issues a single-use reset token. Always returns ok=true (no user "
    "enumeration).",
)
async def request_password_reset(
    body: ResetPasswordRequest,
    request: Request,
    service: AuthService = Depends(get_enterprise_auth_service),
) -> dict[str, Any]:
    result = await service.request_password_reset(email=body.email, ip=_client(request)["ip"])
    return result


@router.post(
    "/password/reset/confirm",
    summary="Complete the password reset",
    description="Consumes the reset token and sets a new password.",
)
async def reset_password(
    body: ResetPasswordConfirmRequest,
    request: Request,
    service: AuthService = Depends(get_enterprise_auth_service),
) -> dict[str, str]:
    await service.reset_password(
        token=body.token, new_password=body.new_password, ip=_client(request)["ip"]
    )
    return {"message": "password reset successful"}


@router.post(
    "/verify-email/request",
    summary="Request an email verification token",
)
async def request_email_verification(
    user: Any = Depends(get_current_user),
    service: AuthService = Depends(get_enterprise_auth_service),
) -> dict[str, Any]:
    return await service.request_email_verification(user=user)


@router.post(
    "/verify-email/confirm",
    response_model=VerifyEmailResponse,
    summary="Confirm the email address",
)
async def verify_email(
    body: VerifyEmailRequest,
    request: Request,
    service: AuthService = Depends(get_enterprise_auth_service),
) -> VerifyEmailResponse:
    user = await service.verify_email(token=body.token, ip=_client(request)["ip"])
    return VerifyEmailResponse(email=user.email)


@router.get(
    "/sessions",
    response_model=list[SessionResponse],
    summary="List active sessions",
    description="All active sessions for the authenticated user.",
)
async def list_sessions(
    user: Any = Depends(get_current_user),
    service: AuthService = Depends(get_enterprise_auth_service),
) -> list[SessionResponse]:
    return [SessionResponse(**s) for s in await service.list_sessions(user.id)]


@router.delete(
    "/sessions/{session_id}",
    summary="Revoke a session",
    description="Revoke a specific session owned by the authenticated user.",
)
async def revoke_session(
    session_id: str,
    user: Any = Depends(get_current_user),
    service: AuthService = Depends(get_enterprise_auth_service),
) -> dict[str, Any]:
    revoked = await service.revoke_session(user.id, session_id)
    if not revoked:
        return {"revoked": False, "message": "session not found"}
    return {"revoked": True, "message": "session revoked"}


@router.delete(
    "/sessions",
    summary="Revoke all other sessions",
    description="Revoke every active session except the current one.",
)
async def revoke_all_other_sessions(
    request: Request,
    user: Any = Depends(get_current_user),
    service: AuthService = Depends(get_enterprise_auth_service),
) -> dict[str, Any]:
    current = request.headers.get("x-session-id")
    count = await service.revoke_all_sessions(user.id, except_session_id=current)
    return {"revoked": count}


# ---------------------------------------------------------------------- #
# User model helper (kept out of the public schema module)
# ---------------------------------------------------------------------- #
def _session_user(user: User) -> User:  # pragma: no cover - helper
    return user
