"""Async SQLAlchemy 2 database wiring.

* :class:`Base` — declarative base for all ORM models.
* :class:`Database` — async engine + session factory (aiosqlite for dev/tests,
  asyncpg for PostgreSQL in production).
* :func:`create_all` — creates tables for dev/test (Alembic migrations belong
  to a later phase).
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from .config import DatabaseSettings


class Base(DeclarativeBase):
    """Shared declarative base for every ORM model."""


class Database:
    """Owns the async engine and session factory."""

    def __init__(self, settings: DatabaseSettings) -> None:
        self.settings = settings
        kwargs: dict = {"echo": settings.echo, "future": True}
        if settings.url.startswith("postgresql"):
            kwargs.update(
                pool_size=settings.pool_size,
                max_overflow=settings.max_overflow,
                pool_recycle=settings.pool_recycle_seconds,
                pool_pre_ping=True,
            )
        self.engine: AsyncEngine = create_async_engine(settings.url, **kwargs)
        self.session_factory = async_sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

    async def connect(self) -> None:
        await self.engine.dispose()  # ensure a clean pool on re-init

    async def close(self) -> None:
        await self.engine.dispose()

    async def create_all(self) -> None:
        """Create tables (dev/test only — prod uses Alembic migrations)."""
        import app.models  # noqa: F401  (imports every model into Base.metadata)
        import database.models  # noqa: F401  (Phase 10 enterprise models)

        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def drop_all(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self.session_factory() as session:
            yield session
