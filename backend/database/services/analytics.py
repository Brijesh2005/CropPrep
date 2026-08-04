"""Analytics service: dashboard aggregates for the admin layer."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from database.repositories import (
    FeedbackRepository,
    NotificationRepository,
    PredictionRepository,
    UserRepository,
)
from database.services.redis_store import RedisStore


class AnalyticsService:
    """Aggregated statistics over predictions, users, feedback and activity."""

    def __init__(
        self,
        predictions: PredictionRepository,
        users: UserRepository,
        feedback: FeedbackRepository,
        notifications: NotificationRepository,
        store: RedisStore,
    ) -> None:
        self._predictions = predictions
        self._users = users
        self._feedback = feedback
        self._notifications = notifications
        self._store = store

    async def dashboard(self) -> dict[str, Any]:
        cache_key = "analytics:dashboard"
        cached = await self._store.get(cache_key)
        if cached is not None:
            return cached
        report = {
            "totals": await self._totals(),
            "by_crop": await self._predictions.analytics_by_crop(),
            "by_region": await self._predictions.analytics_by_region(),
            "confidence": await self._predictions.confidence_distribution(),
            "feedback": {
                "average_rating": await self._feedback.average_rating(),
                "count": await self._feedback.count_rows(),
            },
        }
        await self._store.set(cache_key, report, ttl=300)
        return report

    async def _totals(self) -> dict[str, Any]:
        from datetime import timedelta

        now = datetime.utcnow()
        return {
            "users": await self._users.count_rows(),
            "predictions": await self._predictions.count_rows(),
            "predictions_24h": await self._predictions.count_rows(
                self._predictions.model.created_at >= now - timedelta(hours=24)
            ),
            "predictions_7d": await self._predictions.count_rows(
                self._predictions.model.created_at >= now - timedelta(days=7)
            ),
            "feedback": await self._feedback.count_rows(),
            "unread_notifications": await self._notifications.count_rows(
                self._notifications.model.status != "read"
            ),
        }

    async def by_season_year(
        self, *, season: str | None = None, year: int | None = None
    ) -> dict[str, Any]:
        return {
            "by_crop": await self._predictions.analytics_by_crop(season=season, year=year),
            "by_region": await self._predictions.analytics_by_region(season=season, year=year),
        }

    async def invalidate(self) -> None:
        await self._store.delete("analytics:dashboard")
