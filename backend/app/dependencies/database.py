"""Request-scoped database session + repository dependencies."""

from __future__ import annotations

from typing import AsyncIterator, Type, TypeVar

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.base import BaseRepository

R = TypeVar("R", bound=BaseRepository)


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    """Yield a request-scoped async session from the database singleton."""
    database = getattr(request.app.state, "database", None)
    if database is None:
        raise RuntimeError("database is not initialised")
    async with database.session_factory() as session:
        yield session


def get_repository(repo_type: Type[R]):
    """Build a repository bound to the request's session."""

    async def _dependency(session: AsyncSession = Depends(get_session)) -> R:
        return repo_type(session)

    return _dependency
