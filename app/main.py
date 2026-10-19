"""FastAPI application entry point."""

import ipaddress
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.api.dependencies import ApiServices
from app.api.routes.documents import router as documents_router
from app.api.routes.query import router as query_router
from app.core.config import Settings, get_settings
from app.core.errors import ErrorDetail, ErrorResponse, register_error_handlers
from app.core.security import AuthenticationError
from app.observability import Observability, get_observability
from app.observability.metrics import PROMETHEUS_CONTENT_TYPE


class LiveResponse(BaseModel):
    """Liveness response that never depends on external services."""

    status: Literal["ok"] = "ok"


def create_app(
    settings: Settings | None = None,
    *,
    api_services: ApiServices | None = None,
    observability: Observability | None = None,
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
    application.state.observability = observability or get_observability()
    register_error_handlers(application)

    @application.middleware("http")
    async def verified_identity_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        trace_id = str(uuid4())
        request.state.trace_id = trace_id
        telemetry = application.state.observability
        assert isinstance(telemetry, Observability)
        response: Response
        with (
            telemetry.trace_scope(trace_id),
            telemetry.observe(
                component="http",
                operation="request",
                trace_id=trace_id,
                attributes={"http.request.method": request.method},
            ) as observed,
        ):
            if request.url.path.startswith("/v1/") or request.url.path == "/v1":
                services = application.state.api_services
                if not isinstance(services, ApiServices):
                    response = _error_response(
                        503,
                        "service_unavailable",
                        "API service dependencies are not configured",
                    )
                else:
                    try:
                        request.state.principal = services.authenticator.verify_bearer(
                            request.headers.get("Authorization")
                        )
                    except AuthenticationError:
                        response = _error_response(
                            401, "unauthorized", "Invalid or missing bearer token"
                        )
                    else:
                        response = await call_next(request)
            else:
                response = await call_next(request)
            route = getattr(request.scope.get("route"), "path", "unmatched")
            observed.set_attribute("http.route", str(route))
            observed.set_attribute("http.response.status_code", response.status_code)
            observed.set_log_field("http_method", request.method)
            observed.set_log_field("http_route", str(route))
            observed.set_log_field("http_status_code", response.status_code)
            response.headers["X-Trace-ID"] = trace_id
            return response

    @application.get("/health/live", response_model=LiveResponse, tags=["health"])
    async def health_live() -> LiveResponse:
        return LiveResponse()

    @application.get("/metrics", include_in_schema=False)
    async def metrics(request: Request) -> Response:
        if not _metrics_access_allowed(request, application):
            return _error_response(403, "forbidden", "Metrics access is restricted")
        telemetry = application.state.observability
        assert isinstance(telemetry, Observability)
        return Response(
            content=telemetry.metrics.render(),
            media_type=PROMETHEUS_CONTENT_TYPE,
        )

    application.include_router(documents_router)
    application.include_router(query_router)
    static_directory = Path(__file__).with_name("static")
    if static_directory.is_dir():
        application.mount("/", StaticFiles(directory=static_directory, html=True), name="demo")
    return application


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    payload = ErrorResponse(error=ErrorDetail(code=code, message=message)).model_dump(
        exclude_none=True
    )
    return JSONResponse(status_code=status_code, content=payload)


def _metrics_access_allowed(request: Request, application: FastAPI) -> bool:
    client = request.client
    if client is not None:
        try:
            if ipaddress.ip_address(client.host).is_loopback:
                return True
        except ValueError:
            pass
    services = application.state.api_services
    if not isinstance(services, ApiServices):
        return False
    try:
        principal = services.authenticator.verify_bearer(request.headers.get("Authorization"))
    except AuthenticationError:
        return False
    return "metrics:read" in principal.scopes


def create_runtime_app() -> FastAPI:
    """Assemble only the explicitly selected runtime; real mode never falls back."""
    settings = get_settings()
    services = None
    if settings.app_mode == "fake" and settings.app_env == "local":
        from app.demo import build_demo_services

        services = build_demo_services(settings)
    return create_app(settings, api_services=services)


app = create_runtime_app()
