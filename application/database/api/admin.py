"""Enterprise admin routes: analytics dashboard and audit trail.

Complement the Phase 8 ``/admin`` routes (``/admin/dashboard``,
``/admin/statistics``, ``/admin/retrain``).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, Query

from app.core.security import ROLE_ADMIN
from app.dependencies.enterprise import get_analytics_service, get_audit_service
from app.dependencies.security import require_role
from database.services.analytics import AnalyticsService
from database.services.audit_service import AuditService

router = APIRouter(prefix="/admin", tags=["admin-enterprise"])


@router.get(
    "/enterprise/dashboard",
    summary="Enterprise analytics dashboard",
    description="Aggregated platform analytics (totals, crops, regions, feedback).",
)
async def enterprise_dashboard(
    _: Any = Depends(require_role(ROLE_ADMIN)),
    service: AnalyticsService = Depends(get_analytics_service),
) -> dict[str, Any]:
    return await service.dashboard()


@router.get(
    "/enterprise/analytics",
    summary="Analytics by season and year",
)
async def enterprise_analytics(
    season: str | None = Query(default=None),
    year: int | None = Query(default=None),
    _: Any = Depends(require_role(ROLE_ADMIN)),
    service: AnalyticsService = Depends(get_analytics_service),
) -> dict[str, Any]:
    return await service.by_season_year(season=season, year=year)


@router.get(
    "/enterprise/audit",
    summary="Query the audit trail",
    description="Filtered, paginated audit log (admin only).",
)
async def audit_log(
    user_id: int | None = Query(default=None),
    action: str | None = Query(default=None),
    resource_type: str | None = Query(default=None),
    outcome: str | None = Query(default=None),
    date_from: datetime | None = Query(default=None),
    date_to: datetime | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    _: Any = Depends(require_role(ROLE_ADMIN)),
    service: AuditService = Depends(get_audit_service),
) -> dict[str, Any]:
    return await service.query(
        user_id=user_id,
        action=action,
        resource_type=resource_type,
        outcome=outcome,
        date_from=date_from,
        date_to=date_to,
        limit=limit,
        offset=offset,
    )
