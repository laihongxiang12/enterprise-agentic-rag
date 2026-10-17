"""FastAPI application entry point."""

from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel

from app.core.config import Settings, get_settings
from app.core.errors import register_error_handlers


class LiveResponse(BaseModel):
    """Liveness response that never depends on external services."""

    status: Literal["ok"] = "ok"


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create an application with explicitly injectable settings."""
    resolved_settings = settings or get_settings()
    application = FastAPI(
        title=resolved_settings.app_name,
        version="0.1.0",
        docs_url="/docs" if resolved_settings.app_env != "production" else None,
        redoc_url=None,
    )
    application.state.settings = resolved_settings
    register_error_handlers(application)

    @application.get("/health/live", response_model=LiveResponse, tags=["health"])
    async def health_live() -> LiveResponse:
        return LiveResponse()

    return application


app = create_app()
