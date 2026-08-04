"""Request-ID middleware — assigns / propagates a correlation ID."""

from __future__ import annotations

import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.logging import set_correlation_id


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Set ``X-Request-ID`` (propagating the client's when present)."""

    def __init__(self, app, header: str = "X-Request-ID") -> None:
        super().__init__(app)
        self.header = header

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get(self.header) or str(uuid.uuid4())
        set_correlation_id(request_id)
        response: Response = await call_next(request)
        response.headers[self.header] = request_id
        return response
