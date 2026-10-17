"""Real Milvus upsert, server-side authorization, version, and delete tests."""

import asyncio
from uuid import uuid4

import pytest

from app.core.config import Settings
from app.retrieval.base import VectorStore
from app.schemas.retrieval import AuthFilter, RetrievedChunk
from app.stores.milvus import MilvusVectorStore

pytestmark = pytest.mark.integration


def chunk(
    chunk_id: str,
    content: str,
    *,
    tenant_id: str,
    document_id: str,
    visibility: str = "restricted",
    groups: list[str] | None = None,
    document_version: int = 1,
) -> RetrievedChunk:
    return RetrievedChunk.model_validate(
        {
            "tenant_id": tenant_id,
            "workspace_id": "hr",
            "document_id": document_id,
            "document_version": document_version,
            "chunk_id": chunk_id,
            "parent_id": f"parent-{chunk_id}",
            "content": content,
            "title": "Employee policy",
            "source_uri": f"fixture:///{tenant_id}/{document_id}",
            "visibility": visibility,
            "acl_groups": groups or [],
            "heading_path": ["Policies"],
        }
    )


@pytest.mark.asyncio
async def test_real_milvus_server_side_acl_version_delete_and_dimension() -> None:
    uri = Settings().test_milvus_uri
    if uri is None:
        pytest.skip("TEST_MILVUS_URI is not configured for real Milvus integration")

    suffix = uuid4().hex[:12]
    collection = f"stage07_{suffix}"
    active_index = f"active-{suffix}"
    stale_index = f"stale-{suffix}"
    tenant_a = f'tenant-a-"-{suffix}'
    tenant_b = f"tenant-b-{suffix}"
    store = MilvusVectorStore(uri, collection, dimension=4)
    active = chunk(
        "active",
        "Annual leave rules",
        tenant_id=tenant_a,
        document_id="leave",
        groups=["hr"],
    )
    stale = chunk(
        "stale",
        "Old annual leave rules",
        tenant_id=tenant_a,
        document_id="leave",
        groups=["hr"],
    )
    finance = chunk(
        "finance",
        "Confidential compensation policy",
        tenant_id=tenant_a,
        document_id="finance",
        groups=["finance"],
    )
    public = chunk(
        "public",
        "Company handbook",
        tenant_id=tenant_a,
        document_id="handbook",
        visibility="tenant",
    )
    other_tenant = chunk(
        "other-tenant",
        "Annual leave rules",
        tenant_id=tenant_b,
        document_id="leave",
        groups=["hr"],
    )
    other_workspace = active.model_copy(
        update={"workspace_id": "finance", "chunk_id": "other-workspace"}
    )
    wrong_version = active.model_copy(
        update={"document_version": 2, "chunk_id": "wrong-version"}
    )
    try:
        await store.initialize()
        assert isinstance(store, VectorStore)
        await store.upsert(
            [active, finance, public, other_tenant, other_workspace, wrong_version],
            [
                [0.8, 0.2, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.9, 0.1, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
            active_index,
        )
        await store.upsert([stale], [[1.0, 0.0, 0.0, 0.0]], stale_index)
        pending = store._row(
            active.model_copy(update={"chunk_id": "pending"}),
            [1.0, 0.0, 0.0, 0.0], active_index,
        )
        pending["ready"] = False
        await asyncio.to_thread(
            store._client.upsert, collection_name=collection, data=[pending], timeout=15,
        )

        scope = AuthFilter(
            tenant_id=tenant_a,
            workspace_id="hr",
            group_ids=frozenset({"hr"}),
            allowed_document_ids=frozenset({"leave", "finance", "handbook"}),
            allowed_index_versions=frozenset({active_index}),
            allowed_document_versions=frozenset(
                (doc, 1, active_index) for doc in ("leave", "finance", "handbook")
            ),
        )
        results = await store.search_dense([1.0, 0.0, 0.0, 0.0], scope, top_k=10)
        assert {item.chunk_id for item in results} == {"active", "public"}
        assert all(item.tenant_id == tenant_a for item in results)
        assert "finance" not in {item.chunk_id for item in results}
        assert "other-tenant" not in {item.chunk_id for item in results}
        assert "stale" not in {item.chunk_id for item in results}
        assert all(item.index_version == active_index for item in results)
        assert [item.chunk_id for item in await store.search_dense(
            [1.0, 0.0, 0.0, 0.0], scope, top_k=1
        )] == ["public"]

        # Another document may use the stale index label; it must not authorize
        # that label on leave. Independent IN lists would admit the stale row.
        mixed_scope = scope.model_copy(update={
            "allowed_index_versions": frozenset({active_index, stale_index}),
            "allowed_document_versions": scope.allowed_document_versions
            | frozenset({("handbook", 1, stale_index)}),
        })
        assert {item.chunk_id for item in await store.search_dense(
            [1.0, 0.0, 0.0, 0.0], mixed_scope, top_k=10
        )} == {"active", "public"}

        updated = active.model_copy(update={"content": "Updated annual leave rules"})
        await store.upsert([updated], [[0.8, 0.2, 0.0, 0.0]], active_index)
        updated_results = await store.search_dense([1.0, 0.0, 0.0, 0.0], scope, top_k=10)
        assert len(updated_results) == 2
        assert next(item for item in updated_results if item.chunk_id == "active").content == (
            "Updated annual leave rules"
        )

        incompatible = MilvusVectorStore(uri, collection, dimension=8)
        try:
            with pytest.raises(ValueError, match="dimension"):
                await incompatible.initialize()
        finally:
            await incompatible.close()

        no_groups = scope.model_copy(update={"group_ids": frozenset()})
        assert [
            item.chunk_id
            for item in await store.search_dense(
                [1.0, 0.0, 0.0, 0.0], no_groups, top_k=10
            )
        ] == ["public"]

        stale_scope = scope.model_copy(
            update={
                "allowed_document_ids": frozenset({"leave"}),
                "allowed_index_versions": frozenset({stale_index}),
                "allowed_document_versions": frozenset({("leave", 1, stale_index)}),
            }
        )
        assert [
            item.chunk_id
            for item in await store.search_dense(
                [1.0, 0.0, 0.0, 0.0], stale_scope, top_k=10
            )
        ] == ["stale"]

        with pytest.raises(ValueError, match="embedding dimension"):
            await store.search_dense([1.0, 0.0], scope, top_k=3)
        with pytest.raises(ValueError, match="embedding dimension"):
            await store.upsert([active], [[1.0, 0.0]], active_index)

        await store.delete_version(tenant_a, "hr", "leave", 1, active_index)
        remaining = await store.search_dense([1.0, 0.0, 0.0, 0.0], scope, top_k=10)
        assert [item.chunk_id for item in remaining] == ["public"]
        other_scope = scope.model_copy(update={"tenant_id": tenant_b})
        assert [item.chunk_id for item in await store.search_dense(
            [1.0, 0.0, 0.0, 0.0], other_scope, top_k=10
        )] == ["other-tenant"]
        assert [
            item.chunk_id
            for item in await store.search_dense(
                [1.0, 0.0, 0.0, 0.0], stale_scope, top_k=10
            )
        ] == ["stale"]
    finally:
        await store.drop_collection()
        await store.close()
