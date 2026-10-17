"""Offline tests for the stage-01 application shell."""

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from app.core.config import Settings
from app.main import create_app


@pytest.mark.asyncio
async def test_health_live_is_independent_of_external_services() -> None:
    settings = Settings(app_env="test", app_mode="fake", _env_file=None)
    application = create_app(settings)

    async with AsyncClient(
        transport=ASGITransport(app=application),
        base_url="http://testserver",
    ) as client:
        response = await client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_settings_are_injectable_without_network_access() -> None:
    settings = Settings(
        app_name="Injected RAG",
        app_env="test",
        app_mode="local",
        _env_file=None,
    )
    application = create_app(settings)

    assert application.title == "Injected RAG"
    assert application.state.settings is settings


def test_real_mode_rejects_missing_llm_key() -> None:
    with pytest.raises(ValidationError, match="LLM_API_KEY is required"):
        Settings(app_mode="real", llm_api_key=None, _env_file=None)
