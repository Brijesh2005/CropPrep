"""Users routes: profile read + update."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from app.dependencies.security import get_current_user
from app.modules.users.dependencies import get_user_service
from app.modules.users.schemas import ProfileUpdateRequest, UserResponse
from app.modules.users.service import UserService

router = APIRouter(prefix="/users", tags=["users"])


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Current user profile",
    description="Return the authenticated user's profile.",
)
async def get_me(user: Any = Depends(get_current_user)) -> UserResponse:
    return UserResponse(**user.to_dict())


@router.put(
    "/profile",
    response_model=UserResponse,
    summary="Update the current user's profile",
    description="Update the full name and/or password of the authenticated user.",
)
async def update_profile(
    body: ProfileUpdateRequest,
    user: Any = Depends(get_current_user),
    service: UserService = Depends(get_user_service),
) -> UserResponse:
    updated = await service.update_profile(
        user.id, full_name=body.full_name, password=body.password
    )
    return UserResponse(**updated.to_dict())
