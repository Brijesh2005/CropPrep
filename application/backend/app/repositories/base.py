"""Generic async repository base."""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

ModelT = TypeVar("ModelT")


class BaseRepository(Generic[ModelT]):
    """Thin, generic async CRUD over one SQLAlchemy model."""

    model: type[ModelT]

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get(self, obj_id: int) -> ModelT | None:
        return await self.session.get(self.model, obj_id)

    async def list(self, *, limit: int = 100, offset: int = 0) -> list[ModelT]:
        result = await self.session.execute(
            select(self.model).limit(limit).offset(offset)
        )
        return list(result.scalars().all())

    async def add(self, obj: ModelT) -> ModelT:
        self.session.add(obj)
        await self.session.flush()
        return obj

    async def delete(self, obj: ModelT) -> None:
        await self.session.delete(obj)

    async def commit(self) -> None:
        await self.session.commit()

    async def refresh(self, obj: ModelT) -> ModelT:
        await self.session.refresh(obj)
        return obj

    async def _scalars(self, statement: Any) -> list[ModelT]:
        result = await self.session.execute(statement)
        return list(result.scalars().all())
