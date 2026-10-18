"""In-process HTTP contract tests; ASGITransport performs no network I/O."""

from __future__ import annotations

import json
from dataclasses import dataclass

import httpx
import pytest
from pydantic import SecretStr

from app.api.dependencies import ApiServices, NoopRequestLimiter
from app.core.config import Settings
from app.core.dependencies import VerifiedIdentity
from app.core.errors import AppError
from app.core.security import DevTokenService
from app.main import create_app
from app.schemas.api import Citation, QueryRequest, QueryResponse, Usage
from app.schemas.documents import Principal
from app.services.document_service import InMemoryDocumentService, MemoryIngestQueue

SECRET = "stage-15-local-secret-that-is-at-least-32-characters"


class FakeQueryService:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.principals: list[Principal] = []

    async def query(
        self,
        request: QueryRequest,
        principal: Principal,
        *,
        trace_id: str,
    ) -> QueryResponse:
        self.principals.append(principal)
        if self.fail:
            raise AppError(code="query_failed", message="Fake query failed", status_code=503)
        return QueryResponse(
            trace_id=trace_id,
            answer=f"依据授权制度, {request.query} [1]",
            citations=[
                Citation(
                    document_id="doc-1",
                    document_version=1,
                    chunk_id="chunk-1",
                    title="员工手册",
                    source_uri="fixture:///handbook.md",
                    snippet="授权制度证据",
                )
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            latency_ms=3,
            retries=0,
            finish_reason="completed",
        )


class DenyLimiter:
    async def check(self, principal: Principal, operation: str) -> None:
        del principal, operation
        raise AppError(code="rate_limited", message="Too many requests", status_code=429)


@dataclass
class ApiHarness:
    client: httpx.AsyncClient
    token: str
    query_service: FakeQueryService
    queue: MemoryIngestQueue


def settings() -> Settings:
    return Settings(
        app_env="local",
        app_mode="fake",
        bind_host="127.0.0.1",
        dev_auth_enabled=True,
        dev_auth_secret=SecretStr(SECRET),
        upload_max_bytes=32,
    )


async def harness(
    *,
    scopes: tuple[str, ...] = ("documents:read", "documents:write", "query:read"),
    member: bool = True,
    fail_query: bool = False,
    limited: bool = False,
) -> ApiHarness:
    configured = settings()
    authenticator = DevTokenService(configured)
    identity = VerifiedIdentity(
        user_id="alice",
        tenant_id="tenant-a",
        group_ids=("hr",),
        scopes=scopes,
    )
    token = authenticator.issue(identity)
    queue = MemoryIngestQueue()
    memberships = {("tenant-a", "hr", "alice")} if member else set()
    documents = InMemoryDocumentService(
        queue,
        workspace_memberships=memberships,
        max_bytes=configured.upload_max_bytes,
    )
    queries = FakeQueryService(fail=fail_query)
    services = ApiServices(
        authenticator=authenticator,
        documents=documents,
        queries=queries,
        limiter=DenyLimiter() if limited else NoopRequestLimiter(),
        max_upload_bytes=configured.upload_max_bytes,
    )
    app = create_app(configured, api_services=services)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    )
    return ApiHarness(client=client, token=token, query_service=queries, queue=queue)


def auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_authentication_is_required() -> None:
    api = await harness()
    async with api.client:
        response = await api.client.post("/v1/query", json={"query": "制度是什么?"})
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


@pytest.mark.asyncio
async def test_document_upload_status_idempotency_and_delete() -> None:
    api = await harness()
    request = {
        "data": {"workspace_id": "hr", "visibility": "restricted", "acl_groups": "hr"},
        "files": {"file": ("policy.md", b"# Policy\nannual leave", "text/markdown")},
        "headers": {**auth(api.token), "Idempotency-Key": "upload-1"},
    }
    async with api.client:
        first = await api.client.post("/v1/documents", **request)
        repeated = await api.client.post("/v1/documents", **request)
        document_id = first.json()["document_id"]
        current = await api.client.get(
            f"/v1/documents/{document_id}",
            params={"workspace_id": "hr"},
            headers=auth(api.token),
        )
        deleted = await api.client.delete(
            f"/v1/documents/{document_id}",
            params={"workspace_id": "hr"},
            headers=auth(api.token),
        )
        missing = await api.client.get(
            f"/v1/documents/{document_id}",
            params={"workspace_id": "hr"},
            headers=auth(api.token),
        )
    assert first.status_code == 202
    assert repeated.json() == first.json()
    assert len(api.queue.items) == 1
    assert current.json()["status"] == "pending"
    assert deleted.json() == {"document_id": document_id, "status": "deleted"}
    assert missing.status_code == 404


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "expected"),
    [("scope", 403), ("workspace", 403), ("size", 413), ("mime", 422)],
)
async def test_document_error_semantics(case: str, expected: int) -> None:
    api = await harness(
        scopes=("documents:read",) if case == "scope" else ("documents:read", "documents:write"),
        member=case != "workspace",
    )
    content = b"x" * 33 if case == "size" else b"# valid"
    filename = "policy.exe" if case == "mime" else "policy.md"
    content_type = "application/octet-stream" if case == "mime" else "text/markdown"
    async with api.client:
        response = await api.client.post(
            "/v1/documents",
            data={"workspace_id": "hr", "visibility": "restricted", "acl_groups": "hr"},
            files={"file": (filename, content, content_type)},
            headers=auth(api.token),
        )
    assert response.status_code == expected


@pytest.mark.asyncio
async def test_query_contract_rejects_body_tenant_and_handles_rate_limit() -> None:
    normal = await harness()
    limited = await harness(limited=True)
    async with normal.client, limited.client:
        invalid = await normal.client.post(
            "/v1/query",
            json={"query": "制度", "tenant_id": "tenant-b"},
            headers=auth(normal.token),
        )
        throttled = await limited.client.post(
            "/v1/query",
            json={"query": "制度"},
            headers=auth(limited.token),
        )
    assert invalid.status_code == 422
    assert invalid.json()["error"]["code"] == "validation_error"
    assert throttled.status_code == 429
    assert throttled.json()["error"]["code"] == "rate_limited"


@pytest.mark.asyncio
async def test_query_happy_path_uses_verified_principal() -> None:
    api = await harness()
    async with api.client:
        response = await api.client.post(
            "/v1/query",
            json={"query": "年假制度", "workspace_id": "hr"},
            headers=auth(api.token),
        )
    assert response.status_code == 200
    assert response.json()["trace_id"]
    assert response.json()["citations"][0]["document_id"] == "doc-1"
    assert api.query_service.principals[0].tenant_id == "tenant-a"


@pytest.mark.asyncio
async def test_sse_is_parseable_and_done_exactly_once() -> None:
    api = await harness()
    async with api.client:
        response = await api.client.post(
            "/v1/query/stream",
            json={"query": "年假制度", "workspace_id": "hr"},
            headers=auth(api.token),
        )
    events = parse_sse(response.text)
    names = [name for name, _ in events]
    assert response.headers["content-type"].startswith("text/event-stream")
    assert names[0:2] == ["trace", "status"]
    assert "delta" in names and "citations" in names
    assert names.count("done") == 1
    assert "error" not in names
    assert events[1][1]["streaming_mode"] == "simulated_final_answer_segments"
    assert events[1][1]["token_level"] is False


@pytest.mark.asyncio
async def test_sse_error_has_no_done_event() -> None:
    api = await harness(fail_query=True)
    async with api.client:
        response = await api.client.post(
            "/v1/query/stream",
            json={"query": "失败路径"},
            headers=auth(api.token),
        )
    events = parse_sse(response.text)
    assert [name for name, _ in events].count("error") == 1
    assert all(name != "done" for name, _ in events)
    assert events[-1][1]["code"] == "query_failed"


def parse_sse(body: str) -> list[tuple[str, dict[str, object]]]:
    parsed: list[tuple[str, dict[str, object]]] = []
    for block in body.strip().split("\n\n"):
        lines = block.splitlines()
        assert lines[0].startswith("event: ")
        assert lines[1].startswith("data: ")
        payload = json.loads(lines[1].removeprefix("data: "))
        assert isinstance(payload, dict)
        parsed.append((lines[0].removeprefix("event: "), payload))
    return parsed
