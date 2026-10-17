"""Stable application error contracts."""

from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel


class ErrorDetail(BaseModel):
    """Safe error payload returned by the HTTP layer."""

    code: str
    message: str
    trace_id: str | None = None


class ErrorResponse(BaseModel):
    """Top-level error response schema."""

    error: ErrorDetail


class AppError(Exception):
    """Expected application failure with an HTTP-safe message."""

    def __init__(self, *, code: str, message: str, status_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def register_error_handlers(app: FastAPI) -> None:
    """Register handlers for expected application failures."""

    @app.exception_handler(AppError)
    async def handle_app_error(_request: Request, exc: AppError) -> JSONResponse:
        payload: dict[str, Any] = ErrorResponse(
            error=ErrorDetail(code=exc.code, message=exc.message)
        ).model_dump(exclude_none=True)
        return JSONResponse(status_code=exc.status_code, content=payload)
