"""Offline validation and dependency-failure checks for the Milvus boundary."""

from unittest.mock import Mock

import pytest

from app.schemas.retrieval import AuthFilter, RetrievedChunk
from app.stores.milvus import MilvusVectorStore


def scope() -> AuthFilter:
    return AuthFilter(
        tenant_id="a", workspace_id="hr", group_ids=frozenset({"hr"}),
        allowed_document_ids=frozenset({"leave"}),
        allowed_index_versions=frozenset({"v1"}),
        allowed_document_versions=frozenset({("leave", 1, "v1")}),
    )


def chunk() -> RetrievedChunk:
    return RetrievedChunk(
        tenant_id="a", workspace_id="hr", document_id="leave", document_version=1,
        chunk_id="c", parent_id="p", content="Annual leave", title="Policy",
        source_uri="fixture:///leave", visibility="restricted", acl_groups=["hr"],
    )


@pytest.mark.parametrize("field", [
    "allowed_document_ids", "allowed_index_versions", "allowed_document_versions",
])
async def test_empty_authorization_never_connects(
    monkeypatch: pytest.MonkeyPatch, field: str,
) -> None:
    client = Mock(side_effect=AssertionError("must not connect"))
    monkeypatch.setattr("app.stores.milvus.MilvusClient", client)
    store = MilvusVectorStore("http://127.0.0.1:19530", "unit", 4)
    assert await store.search_dense([1, 0, 0, 0], scope().model_copy(
        update={field: frozenset()},
    ), 10) == []
    client.assert_not_called()


@pytest.mark.parametrize("groups", [[], ["   "], ["中" * 100]])
async def test_invalid_acl_is_rejected_before_connect(
    monkeypatch: pytest.MonkeyPatch, groups: list[str],
) -> None:
    client = Mock(side_effect=AssertionError("must not connect"))
    monkeypatch.setattr("app.stores.milvus.MilvusClient", client)
    store = MilvusVectorStore("http://127.0.0.1:19530", "unit", 4)
    with pytest.raises(ValueError, match="ACL"):
        await store.upsert([chunk().model_copy(update={"acl_groups": groups})],
                           [[1, 0, 0, 0]], "v1")
    client.assert_not_called()


@pytest.mark.parametrize("vector", [[1, 0], [float("nan"), 0, 0, 0]])
async def test_invalid_vector_never_connects(
    monkeypatch: pytest.MonkeyPatch, vector: list[float],
) -> None:
    client = Mock(side_effect=AssertionError("must not connect"))
    monkeypatch.setattr("app.stores.milvus.MilvusClient", client)
    store = MilvusVectorStore("http://127.0.0.1:19530", "unit", 4)
    with pytest.raises(ValueError, match="embedding"):
        await store.search_dense(vector, scope(), 10)
    client.assert_not_called()


async def test_connection_failure_is_propagated(monkeypatch: pytest.MonkeyPatch) -> None:
    client = Mock(side_effect=ConnectionError("service unavailable"))
    monkeypatch.setattr("app.stores.milvus.MilvusClient", client)
    store = MilvusVectorStore("http://127.0.0.1:19530", "unit", 4, timeout_seconds=0.1)
    with pytest.raises(ConnectionError, match="unavailable"):
        await store.search_dense([1, 0, 0, 0], scope(), 10)
    sent = client.call_args.kwargs
    assert sent["uri"] == "http://127.0.0.1:19530"
    assert sent["timeout"] == 0.1
    assert sent["alias"].startswith("enterprise-rag-")
    await store.close()


async def test_search_timeout_keeps_server_filter_and_propagates() -> None:
    store = MilvusVectorStore("http://127.0.0.1:19530", "unit", 4, timeout_seconds=0.1)
    client = Mock()
    client.search.side_effect = TimeoutError("search deadline")
    store._client = client
    store._initialized = True
    with pytest.raises(TimeoutError, match="deadline"):
        await store.search_dense([1, 0, 0, 0], scope(), 10)
    sent = client.search.call_args.kwargs
    assert sent["timeout"] == 0.1
    assert "ready == true" in sent["filter"]
    assert "ARRAY_CONTAINS_ANY" in sent["filter"]
    assert sent["filter_params"]["doc_0"] == "leave"
    assert sent["filter_params"]["ver_0"] == 1
    assert sent["filter_params"]["idx_0"] == "v1"
    client.search.assert_called_once()
    await store.close()
