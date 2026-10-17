"""Offline Dense/BM25/RRF baseline and authorization-prefilter tests."""

import pytest

from app.retrieval.base import Retriever
from app.retrieval.fakes import FakeEmbedding, FakeVectorStore
from app.retrieval.hybrid import (
    OfflineHybridRetriever,
    reciprocal_rank_fusion,
    tokenize_text,
)
from app.schemas.retrieval import AuthFilter, RetrievedChunk


def chunk(
    chunk_id: str,
    content: str,
    *,
    tenant_id: str = "tenant-a",
    workspace_id: str = "hr",
    document_id: str | None = None,
    document_version: int = 1,
) -> RetrievedChunk:
    resolved_document_id = document_id or f"doc-{chunk_id}"
    return RetrievedChunk(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        document_id=resolved_document_id,
        document_version=document_version,
        chunk_id=chunk_id,
        parent_id=f"parent-{chunk_id}",
        content=content,
        title="员工制度",
        source_uri=f"memory://{tenant_id}/{resolved_document_id}",
    )


def auth_filter(
    *document_ids: str,
    tenant_id: str = "tenant-a",
    index_versions: tuple[str, ...] = ("index-active",),
) -> AuthFilter:
    return AuthFilter(
        tenant_id=tenant_id,
        workspace_id="hr",
        allowed_document_ids=frozenset(document_ids),
        allowed_index_versions=frozenset(index_versions),
    )


def test_chinese_tokenizer_emits_exact_bigrams() -> None:
    tokens = tokenize_text("员工年假申请 Policy-2026")
    assert "年假" in tokens
    assert "申请" in tokens
    assert "policy" in tokens
    assert "2026" in tokens


def test_rrf_matches_verified_legacy_baseline_and_deduplicates() -> None:
    first = chunk("a", "A")
    overlap = chunk("b", "B")
    third = chunk("c", "C")

    results = reciprocal_rank_fusion(
        ([first, overlap, overlap], [overlap, third]), top_k=3, k=60
    )

    assert [item.chunk_id for item in results] == ["b", "a", "c"]
    assert results[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert results[1].score == pytest.approx(1 / 61)
    assert results[2].score == pytest.approx(1 / 62)
    assert overlap.score is None


def test_rrf_top_n_is_deterministic_for_equal_scores() -> None:
    first = chunk("a", "A")
    second = chunk("b", "B")

    results = reciprocal_rank_fusion(([second], [first]), top_k=1, k=60)

    assert [item.chunk_id for item in results] == ["b"]


@pytest.mark.asyncio
async def test_bm25_hits_chinese_term_after_authorization_prefilter() -> None:
    store = FakeVectorStore()
    allowed = chunk("allowed", "员工年假申请需要提前三天", document_id="leave")
    unrelated = chunk("other", "差旅费用报销流程", document_id="expense")
    forbidden = chunk(
        "secret",
        "年假申请秘密审批名单",
        tenant_id="tenant-b",
        document_id="leave",
    )
    await store.upsert([allowed, unrelated, forbidden], [[1.0], [1.0], [1.0]], "index-active")

    results = await store.search_lexical(
        "年假申请", auth_filter("leave", "expense"), top_k=5
    )

    assert [item.chunk_id for item in results] == ["allowed"]
    assert results[0].score is not None and results[0].score > 0


@pytest.mark.asyncio
async def test_hybrid_prefilters_tenant_document_and_index_version_before_scoring() -> None:
    embedding = FakeEmbedding(dimension=128)
    store = FakeVectorStore()
    active = chunk("active", "普通员工手册", document_id="leave")
    stale = chunk("stale", "年假申请旧版规则", document_id="leave")
    forbidden = chunk("forbidden", "年假申请机密规则", document_id="secret")
    other_tenant = chunk(
        "same-name",
        "年假申请另一个租户规则",
        tenant_id="tenant-b",
        document_id="leave",
    )

    semantic_query = "休假权益"
    semantic_vector = await embedding.embed_query(semantic_query)
    await store.upsert([active], [semantic_vector], "index-active")
    await store.upsert(
        [stale], await embedding.embed_documents([stale.content]), "index-stale"
    )
    await store.upsert(
        [forbidden, other_tenant],
        await embedding.embed_documents([forbidden.content, other_tenant.content]),
        "index-active",
    )
    retriever = OfflineHybridRetriever(store, embedding)

    results = await retriever.aretrieve(
        semantic_query, auth_filter("leave"), top_k=5
    )

    assert isinstance(retriever, Retriever)
    assert [item.chunk_id for item in results] == ["active"]
    assert await retriever.aretrieve("", auth_filter("leave"), top_k=5) == []
    assert (
        await retriever.aretrieve(
            "年假申请",
            auth_filter("leave", index_versions=("index-missing",)),
            top_k=5,
        )
        == []
    )


@pytest.mark.asyncio
async def test_hybrid_fails_closed_without_both_allowlists() -> None:
    embedding = FakeEmbedding()
    store = FakeVectorStore()
    retriever = OfflineHybridRetriever(store, embedding)
    missing_documents = AuthFilter(
        tenant_id="tenant-a",
        workspace_id="hr",
        allowed_index_versions=frozenset({"index-active"}),
    )
    missing_versions = AuthFilter(
        tenant_id="tenant-a",
        workspace_id="hr",
        allowed_document_ids=frozenset({"leave"}),
    )

    assert await retriever.aretrieve("年假", missing_documents, top_k=3) == []
    assert await retriever.aretrieve("年假", missing_versions, top_k=3) == []
