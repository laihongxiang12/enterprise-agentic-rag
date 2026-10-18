"""Reproducible black-box negative cases against the ASGI application."""

from __future__ import annotations

from dataclasses import dataclass

import httpx
import pytest
from pydantic import SecretStr

from app.agents.state import AgentRuntimeContext
from app.api.dependencies import ApiServices, NoopRequestLimiter
from app.core.config import Settings
from app.core.dependencies import VerifiedIdentity
from app.core.security import DevTokenService
from app.main import create_app
from app.schemas.documents import Principal
from app.services.document_service import InMemoryDocumentService, MemoryIngestQueue
from app.services.query_service import AgentQueryApplicationService

SECRET = "stage-17-api-secret-that-is-at-least-32-characters"


class NeverContextFactory:
    async def __call__(self, principal: Principal, workspace_id: str | None) -> AgentRuntimeContext:
        del principal, workspace_id
        raise AssertionError("unauthorized requests must not build a runtime context")


@dataclass
class Harness:
    client: httpx.AsyncClient
    authenticator: DevTokenService
    token: str


def configured_settings(*, secret: str = SECRET) -> Settings:
    return Settings(
        app_env="local",
        app_mode="fake",
        bind_host="127.0.0.1",
        dev_auth_enabled=True,
        dev_auth_secret=SecretStr(secret),
        upload_max_bytes=1_024,
    )


def issue(authenticator: DevTokenService, scopes: tuple[str, ...]) -> str:
    return authenticator.issue(
        VerifiedIdentity(
            user_id="alice",
            tenant_id="tenant-a",
            group_ids=("hr",),
            scopes=scopes,
        )
    )


def harness() -> Harness:
    settings = configured_settings()
    authenticator = DevTokenService(settings)
    documents = InMemoryDocumentService(
        MemoryIngestQueue(),
        workspace_memberships={("tenant-a", "hr", "alice")},
        max_bytes=settings.upload_max_bytes,
    )
    services = ApiServices(
        authenticator=authenticator,
        documents=documents,
        queries=AgentQueryApplicationService(NeverContextFactory()),
        limiter=NoopRequestLimiter(),
        max_upload_bytes=settings.upload_max_bytes,
    )
    app = create_app(settings, api_services=services)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://security.test",
    )
    token = issue(
        authenticator,
        ("query:read", "documents:read", "documents:write"),
    )
    return Harness(client=client, authenticator=authenticator, token=token)


def auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_fake_token_is_401_without_secret_or_token_echo() -> None:
    api = harness()
    forged = issue(
        DevTokenService(configured_settings(secret="different-stage-17-secret-long-enough-123456")),
        ("query:read",),
    )
    async with api.client:
        response = await api.client.post(
            "/v1/query", json={"query": "policy"}, headers=auth(forged)
        )
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"
    assert forged not in response.text
    assert "different-stage-17-secret" not in response.text


@pytest.mark.asyncio
async def test_wrong_role_is_403_and_query_body_cannot_supply_tenant() -> None:
    api = harness()
    wrong_role = issue(api.authenticator, ("documents:read",))
    async with api.client:
        forbidden = await api.client.post(
            "/v1/query", json={"query": "policy"}, headers=auth(wrong_role)
        )
        injected_tenant = await api.client.post(
            "/v1/query",
            json={"query": "policy", "tenant_id": "tenant-b"},
            headers=auth(api.token),
        )
    assert forbidden.status_code == 403
    assert forbidden.json()["error"]["code"] == "forbidden"
    assert injected_tenant.status_code == 422


@pytest.mark.asyncio
async def test_deleted_document_cannot_be_read_by_old_identifier() -> None:
    api = harness()
    async with api.client:
        created = await api.client.post(
            "/v1/documents",
            data={"workspace_id": "hr", "visibility": "restricted", "acl_groups": "hr"},
            files={"file": ("policy.md", b"version one", "text/markdown")},
            headers=auth(api.token),
        )
        document_id = created.json()["document_id"]
        deleted = await api.client.delete(
            f"/v1/documents/{document_id}",
            params={"workspace_id": "hr"},
            headers=auth(api.token),
        )
        old_version = await api.client.get(
            f"/v1/documents/{document_id}",
            params={"workspace_id": "hr"},
            headers=auth(api.token),
        )
    assert created.status_code == 202
    assert deleted.status_code == 200
    assert old_version.status_code == 404
    assert "version one" not in old_version.text


@pytest.mark.asyncio
async def test_trace_export_is_not_exposed_and_requires_auth_first() -> None:
    api = harness()
    async with api.client:
        unauthenticated = await api.client.get("/v1/traces/trace-secret")
        authenticated = await api.client.get("/v1/traces/trace-secret", headers=auth(api.token))
    assert unauthenticated.status_code == 401
    assert authenticated.status_code == 404
    assert api.token not in authenticated.text
    assert "trace-secret" not in authenticated.text
