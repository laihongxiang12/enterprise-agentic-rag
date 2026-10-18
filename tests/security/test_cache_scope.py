"""Cache and retrieval boundaries revalidate every authorization dimension."""

import json

import pytest

from app.core.errors import AppError
from app.schemas.retrieval import AuthFilter, RetrievedChunk
from app.services.query_service import CachedRetriever
from app.stores.redis import RetrievalCacheKey


class MemoryCache:
    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}
        self.invalidated: list[str] = []

    async def get(self, key: str) -> bytes | None:
        return self.values.get(key)

    async def set(self, key: str, value: bytes, ttl_seconds: int) -> None:
        assert ttl_seconds > 0
        self.values[key] = value

    async def invalidate(self, key_prefix: str) -> int:
        self.invalidated.append(key_prefix)
        keys = [key for key in self.values if key.startswith(key_prefix)]
        for key in keys:
            self.values.pop(key)
        return len(keys)


class FixedRetriever:
    def __init__(self, result: list[RetrievedChunk]) -> None:
        self.result = result
        self.calls = 0

    async def aretrieve(
        self, query: str, auth_filter: AuthFilter, top_k: int
    ) -> list[RetrievedChunk]:
        del query, auth_filter, top_k
        self.calls += 1
        return self.result


def scope() -> AuthFilter:
    return AuthFilter(
        tenant_id="tenant-a",
        workspace_id="hr",
        group_ids=frozenset({"hr"}),
        allowed_document_ids=frozenset({"policy"}),
        allowed_index_versions=frozenset({"index-v2"}),
        allowed_document_versions=frozenset({("policy", 2, "index-v2")}),
        allowed_acl_versions=frozenset({("policy", 3)}),
    )


def chunk(**updates: object) -> RetrievedChunk:
    values: dict[str, object] = {
        "tenant_id": "tenant-a",
        "workspace_id": "hr",
        "document_id": "policy",
        "document_version": 2,
        "chunk_id": "chunk-2",
        "parent_id": "parent-2",
        "content": "authorized policy",
        "title": "Policy",
        "source_uri": "fixture:///tenant-a/policy.md",
        "visibility": "restricted",
        "acl_groups": ["hr"],
        "index_version": "index-v2",
    }
    values.update(updates)
    return RetrievedChunk.model_validate(values)


def encoded(item: RetrievedChunk) -> bytes:
    return json.dumps([item.model_dump(mode="json")], separators=(",", ":")).encode()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "poison",
    [
        chunk(tenant_id="tenant-b"),
        chunk(workspace_id="finance"),
        chunk(document_id="secret"),
        chunk(document_version=1),
        chunk(index_version="index-deleted"),
        chunk(acl_groups=["finance"]),
    ],
)
async def test_poisoned_or_deleted_version_cache_entry_is_never_returned(
    poison: RetrievedChunk,
) -> None:
    authorization = scope()
    cache = MemoryCache()
    backend = FixedRetriever([chunk()])
    retriever = CachedRetriever(backend, cache, ttl_seconds=60, config_hash="stage17")
    key = RetrievalCacheKey.from_request(authorization, "policy", config_hash="stage17", top_k=8)
    cache.values[key.redis_key] = encoded(poison)

    result = await retriever.aretrieve("policy", authorization, 8)

    assert result == [chunk()]
    assert backend.calls == 1
    assert key.redis_key in cache.invalidated
    assert poison not in result


@pytest.mark.asyncio
async def test_backend_scope_violation_fails_closed_and_is_not_cached() -> None:
    authorization = scope()
    cache = MemoryCache()
    retriever = CachedRetriever(
        FixedRetriever([chunk(tenant_id="tenant-b")]),
        cache,
        ttl_seconds=60,
        config_hash="stage17",
    )
    with pytest.raises(AppError, match="outside the authorization scope") as raised:
        await retriever.aretrieve("policy", authorization, 8)
    assert raised.value.code == "retrieval_scope_violation"
    assert cache.values == {}
