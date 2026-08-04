"""Repository tests (SQLAlchemy async CRUD on SQLite)."""

from __future__ import annotations

import asyncio

from app.core.config import load_settings
from app.core.database import Database
from app.core.security import hash_password
from app.models.prediction import Prediction
from app.models.user import User
from app.repositories.prediction import PredictionRepository
from app.repositories.user import UserRepository


def test_user_repository_crud(tmp_path):
    settings = load_settings(
        env={
            "BACKEND_DATABASE__URL": f"sqlite+aiosqlite:///{tmp_path}/r.db",
            "BACKEND_ENVIRONMENT": "test",
        }
    )

    async def _run():
        db = Database(settings.database)
        await db.create_all()
        async with db.session_factory() as session:
            repo = UserRepository(session)
            user = await repo.add(
                User(email="a@b.c", hashed_password=hash_password("pw"), role="user")
            )
            await session.commit()
            found = await repo.get_by_email("a@b.c")
            assert found is not None and found.id == user.id
            assert await repo.exists("a@b.c") is True
            assert await repo.exists("x@y.z") is False
        await db.close()

    asyncio.run(_run())


def test_prediction_repository(tmp_path):
    settings = load_settings(
        env={
            "BACKEND_DATABASE__URL": f"sqlite+aiosqlite:///{tmp_path}/p.db",
            "BACKEND_ENVIRONMENT": "test",
        }
    )

    async def _run():
        db = Database(settings.database)
        await db.create_all()
        async with db.session_factory() as session:
            repo = PredictionRepository(session)
            await repo.add(
                Prediction(
                    user_id=None, location_lon=74.8, location_lat=13.1,
                    crop="Paddy", crop_probs={"Paddy": 1.0}, confidence=0.9,
                    model_version="v1", source="point",
                )
            )
            await session.commit()
            records = await repo.list()
            assert len(records) == 1
            assert records[0].crop == "Paddy"
            counts = await repo.crop_counts()
            assert counts == [("Paddy", 1)]
        await db.close()

    asyncio.run(_run())
