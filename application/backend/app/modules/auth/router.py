"""Auth routes: register, login (OAuth2 password flow), logout, refresh."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

from app.modules.auth.dependencies import get_auth_service
from app.modules.auth.schemas import AuthUserResponse, RefreshRequest, RegisterRequest, TokenResponse
from app.modules.auth.service import AuthService

router = APIRouter(prefix="/auth", tags=["authentication"])

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


@router.post(
    "/register",
    response_model=AuthUserResponse,
    summary="Register a new user",
    description="Create a user account and return its public profile.",
)
async def register(
    body: RegisterRequest, service: AuthService = Depends(get_auth_service)
) -> AuthUserResponse:
    user = await service.register(
        email=body.email, password=body.password, full_name=body.full_name
    )
    return AuthUserResponse(**user.to_dict())


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Login (OAuth2 password flow)",
    description=(
        "Exchange an email + password for an access token and a refresh token "
        "(application/x-www-form-urlencoded)."
    ),
)
async def login(
    form: Annotated[OAuth2PasswordRequestForm, Depends()],
    service: AuthService = Depends(get_auth_service),
) -> TokenResponse:
    tokens = await service.login(email=form.username, password=form.password)
    return TokenResponse(
        access_token=tokens["access_token"],
        refresh_token=tokens["refresh_token"],
        expires_in=tokens["expires_in"],
    )


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Refresh the access token",
    description="Exchange a valid refresh token for a new token pair.",
)
async def refresh(
    body: RefreshRequest, service: AuthService = Depends(get_auth_service)
) -> TokenResponse:
    tokens = await service.refresh(body.refresh_token)
    return TokenResponse(
        access_token=tokens["access_token"],
        refresh_token=tokens["refresh_token"],
        expires_in=tokens["expires_in"],
    )


@router.post("/logout", summary="Logout (revoke the access token)")
async def logout(
    service: AuthService = Depends(get_auth_service),
    token: str = Depends(oauth2_scheme),
):
    await service.logout(token)
    return {"message": "logged out"}
