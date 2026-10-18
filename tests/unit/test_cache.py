"""Tenant/version-scoped retrieval cache and concurrency tests."""

from __future__ import annotations

import asyncio

import pytest

from app.agents.state import AgentRuntimeContext
from app.core.errors import AppError
from app.core.security import BasicPolicyEngine
from app.llm.fake import FakeLLM
from app.schemas.api import QueryRequest
from app.schemas.documents import Principal
from app.schemas.retrieval import AuthFilter, RetrievedChunk
from app.services.query_service import CachedRetriever, QueryService
from app.stores.checkpoint import checkpoint_config
from app.stores.redis import RetrievalCacheKey


class MemoryCache:
    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}

    async def get(self, key: str) -> bytes | None:
        return self.values.get(key)

    async def set(self, key: str, value: bytes, ttl_seconds: int) -> None:
        assert ttl_seconds > 0
        self.values[key] = value

    async def invalidate(self, key_prefix: str) -> int:
        keys = [key for key in self.values if key.startswith(key_prefix)]
        for key in keys:
            self.values.pop(key)
        return len(keys)


class CountingRetriever:
    def __init__(self) -> None:
        self.calls = 0

    async def aretrieve(
        self, query: str, auth_filter: AuthFilter, top_k: int
    ) -> list[RetrievedChunk]:
        del query, top_k
        self.calls += 1
        return [chunk(auth_filter)]


class SlowGraph:
    def __init__(self, release: asyncio.Event) -> None:
        self.started = asyncio.Event()
        self.release = release

    async def ainvoke(self, state: object, *, config: object, context: object) -> object:
        del config, context
        self.started.set()
        await self.release.wait()
        return state


def auth_filter(
    *,
    tenant: str = "tenant-a",
    group: str = "hr",
    index: str = "index-v1",
    acl_version: int = 1,
) -> AuthFilter:
    return AuthFilter(
        tenant_id=tenant,
        workspace_id="workspace-a",
        group_ids=frozenset({group}),
        allowed_document_ids=frozenset({"doc-1"}),
        allowed_index_versions=frozenset({index}),
        allowed_document_versions=frozenset({("doc-1", 1, index)}),
        allowed_acl_versions=frozenset({("doc-1", acl_version)}),
    )


def chunk(scope: AuthFilter) -> RetrievedChunk:
    return RetrievedChunk(
        tenant_id=scope.tenant_id,
        workspace_id=scope.workspace_id,
        document_id="doc-1",
        document_version=1,
        chunk_id="chunk-1",
        parent_id="parent-1",
        content="annual leave policy",
        title="Policy",
        source_uri="fixture:///policy.md",
        visibility="restricted",
        acl_groups=list(scope.group_ids),
        index_version=next(iter(scope.allowed_index_versions)),
    )


@pytest.mark.asyncio
async def test_cache_hits_only_for_identical_scope_version_and_config() -> None:
    cache = MemoryCache()
    backend = CountingRetriever()
    retriever = CachedRetriever(backend, cache, ttl_seconds=60, config_hash="retrieval-v1")
    first_scope = auth_filter()

    first = await retriever.aretrieve(" Annual   Leave ", first_scope, 8)
    repeated = await retriever.aretrieve("annual leave", first_scope, 8)
    assert repeated == first
    assert backend.calls == 1

    for changed in (
        auth_filter(tenant="tenant-b"),
        auth_filter(group="finance"),
        auth_filter(index="index-v2"),
        auth_filter(acl_version=2),
    ):
        await retriever.aretrieve("annual leave", changed, 8)
    assert backend.calls == 5

    changed_config = CachedRetriever(backend, cache, ttl_seconds=60, config_hash="retrieval-v2")
    await changed_config.aretrieve("annual leave", first_scope, 8)
    assert backend.calls == 6


@pytest.mark.asyncio
async def test_scoped_invalidation_removes_only_one_workspace() -> None:
    cache = MemoryCache()
    backend = CountingRetriever()
    retriever = CachedRetriever(backend, cache, ttl_seconds=60, config_hash="v1")
    first = auth_filter()
    second = first.model_copy(update={"workspace_id": "workspace-b"})
    await retriever.aretrieve("policy", first, 8)
    await retriever.aretrieve("policy", second, 8)

    first_key = RetrievalCacheKey.from_request(first, "policy", config_hash="v1", top_k=8)
    assert await cache.invalidate(first_key.invalidation_prefix) == 1
    assert len(cache.values) == 1


def test_checkpoint_thread_is_bound_to_verified_principal_scope() -> None:
    alice = Principal(
        user_id="alice",
        tenant_id="tenant-a",
        group_ids=["hr"],
        scopes=["query:read"],
    )
    same = checkpoint_config(alice, workspace_id="hr", session_id="session-1")
    assert same == checkpoint_config(alice, workspace_id="hr", session_id="session-1")
    other_tenant = alice.model_copy(update={"tenant_id": "tenant-b"})
    other_group = alice.model_copy(update={"group_ids": ["finance"]})
    assert same != checkpoint_config(other_tenant, workspace_id="hr", session_id="session-1")
    assert same != checkpoint_config(other_group, workspace_id="hr", session_id="session-1")
    assert same != checkpoint_config(alice, workspace_id="finance", session_id="session-1")


@pytest.mark.asyncio
async def test_query_service_rejects_excess_concurrency() -> None:
    principal = Principal(user_id="alice", tenant_id="tenant-a")
    scope = auth_filter()
    context = AgentRuntimeContext(
        principal=principal,
        auth_filter=scope,
        llm=FakeLLM(),
        retriever=CountingRetriever(),
        policy_engine=BasicPolicyEngine(),
    )
    service = QueryService(max_concurrency=1, concurrency_wait_seconds=0.01)
    release = asyncio.Event()
    graph = SlowGraph(release)
    service._graph = graph  # type: ignore[assignment]
    first = asyncio.create_task(
        service.execute(QueryRequest(query="first"), context, trace_id="trace-1")
    )
    await graph.started.wait()
    with pytest.raises(AppError, match="concurrently") as raised:
        await service.execute(QueryRequest(query="second"), context, trace_id="trace-2")
    assert raised.value.status_code == 429
    release.set()
    await first
