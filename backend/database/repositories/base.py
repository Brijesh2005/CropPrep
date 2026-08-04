"""Repository pattern base (Phase 10).

Extends the Phase 8 :class:`app.repositories.base.BaseRepository` with
pagination, counting and statement helpers used across the enterprise
aggregates.
"""

from __future__ import annotations

from typing import Generic, TypeVar

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.base import BaseRepository

ModelT = TypeVar("ModelT")


class DataRepository(BaseRepository, Generic[ModelT]):
    """Async CRUD + pagination for one enterprise aggregate."""

    model: type[ModelT]

    async def paginate(self, statement, *, limit: int, offset: int) -> tuple[list[ModelT], int]:
        """Execute a select and return ``(rows, total_count)``."""
        count_stmt = select(func.count()).select_from(statement.subquery())
        count = int((await self.session.execute(count_stmt)).scalar_one() or 0)
        rows = await self._scalars(
            statement.order_by(self._default_order_by()).limit(limit).offset(offset)
        )
        return rows, count

    def _default_order_by(self):
        return self.model.created_at.desc() if hasattr(self.model, "created_at") else self.model.id.desc()

    async def count_rows(self, *conditions) -> int:
        stmt = select(func.count()).select_from(self.model)
        for condition in conditions:
            stmt = stmt.where(condition)
        return int((await self.session.execute(stmt)).scalar_one() or 0)

    async def save(self, obj: ModelT) -> ModelT:
        await self.add(obj)
        await self.commit()
        await self.refresh(obj)
        return obj
