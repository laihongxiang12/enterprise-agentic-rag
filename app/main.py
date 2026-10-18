"""FastAPI application entry point."""

from collections.abc import Awaitable, Callable
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from app.api.dependencies import ApiServices
from app.api.routes.documents import router as documents_router
from app.api.routes.query import router as query_router
from app.core.config import Settings, get_settings
from app.core.errors import ErrorDetail, ErrorResponse, register_error_handlers
from app.core.security import AuthenticationError


class LiveResponse(BaseModel):
    """Liveness response that never depends on external services."""

    status: Literal["ok"] = "ok"


def create_app(
    settings: Settings | None = None,
    *,
    api_services: ApiServices | None = None,
) -> FastAPI:
    """Create an application with explicitly injectable settings and services."""
    resolved_settings = settings or get_settings()
    application = FastAPI(
        title=resolved_settings.app_name,
        version="0.1.0",
        docs_url="/docs" if resolved_settings.app_env != "production" else None,
        redoc_url=None,
    )
    application.state.settings = resolved_settings
    application.state.api_services = api_services
    register_error_handlers(application)

    @application.middleware("http")
    async def verified_identity_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if request.url.path.startswith("/v1/") or request.url.path == "/v1":
            services = application.state.api_services
            if not isinstance(services, ApiServices):
                return _error_response(
                    503, "service_unavailable", "API service dependencies are not configured"
                )
            try:
                request.state.principal = services.authenticator.verify_bearer(
                    request.headers.get("Authorization")
                )
            except AuthenticationError:
                return _error_response(401, "unauthorized", "Invalid or missing bearer token")
        return await call_next(request)

    @application.get("/health/live", response_model=LiveResponse, tags=["health"])
    async def health_live() -> LiveResponse:
        return LiveResponse()

    application.include_router(documents_router)
    application.include_router(query_router)
    return application


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    payload = ErrorResponse(error=ErrorDetail(code=code, message=message)).model_dump(
        exclude_none=True
    )
    return JSONResponse(status_code=status_code, content=payload)


app = create_app()
