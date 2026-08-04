"""Global exception handlers → structured JSON error responses."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.exceptions import BackendError
from app.core.logging import get_logger, get_correlation_id

logger = get_logger("handlers")


def _payload(code: str, message: str, detail: Any = None, status: int = 500) -> dict[str, Any]:
    return {
        "error": {
            "code": code,
            "message": message,
            "detail": detail,
            "status": status,
            "request_id": get_correlation_id(),
        }
    }


async def backend_error_handler(request: Request, exc: BackendError):
    logger.warning("backend error", code=exc.code, message=exc.message)
    return exc.status_code, _payload(exc.code, exc.message, exc.detail, exc.status_code)


async def validation_error_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    detail = [
        {
            "loc": list(e.get("loc", [])),
            "msg": e.get("msg"),
            "type": e.get("type"),
        }
        for e in errors
    ]
    return 422, _payload("B-VALID-001", "request validation failed", detail, 422)


async def value_error_handler(request: Request, exc: ValueError):
    logger.warning("value error", message=str(exc))
    return 400, _payload("B-VALID-002", str(exc) or "invalid value", None, 400)


async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    code = {401: "B-AUTH-001", 403: "B-AUTH-002", 404: "B-NOTFOUND-001", 429: "B-RATE-001"}.get(
        exc.status_code, f"B-HTTP-{exc.status_code}"
    )
    return exc.status_code, _payload(
        code, str(exc.detail or "HTTP error"), detail=None, status=exc.status_code
    )


async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("unhandled exception", exc_info=exc)
    return 500, _payload("B-ERROR", "internal server error", str(exc), 500)


def register_exception_handlers(app: FastAPI) -> None:
    """Register every handler on ``app`` (responses are returned as tuples to
    keep the handlers simple; Starlette wraps them into JSONResponse)."""

    app.add_exception_handler(BackendError, _wrap(backend_error_handler))
    app.add_exception_handler(RequestValidationError, _wrap(validation_error_handler))
    app.add_exception_handler(ValueError, _wrap(value_error_handler))
    app.add_exception_handler(StarletteHTTPException, _wrap(http_exception_handler))
    app.add_exception_handler(Exception, _wrap(unhandled_exception_handler))


def _wrap(handler):
    """Adapt a handler returning ``(status, payload)`` to Starlette's
    ``(status, payload) -> JSONResponse`` contract."""

    from fastapi.responses import JSONResponse

    async def wrapper(request: Request, exc: Exception):
        status, payload = await handler(request, exc)
        return JSONResponse(status_code=status, content=payload)

    return wrapper
