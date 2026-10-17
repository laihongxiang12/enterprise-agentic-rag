"""Contract validation, security boundaries, and deterministic fake tests."""

from copy import deepcopy

import pytest
from pydantic import ValidationError

from app.core.dependencies import VerifiedIdentity, build_fake_container, build_principal
from app.llm.base import LLMMessage, LLMProvider
from app.llm.fake import FakeLLM
from app.retrieval.base import EmbeddingProvider, VectorStore
from app.retrieval.fakes import FakeEmbedding, FakeVectorStore
from app.schemas.agent import EvidenceGrade, RouteDecision, new_agent_state
from app.schemas.api import Citation, QueryRequest, QueryResponse, Usage
from app.schemas.documents import Principal
from app.schemas.retrieval import AuthFilter, RetrievedChunk


def make_chunk(
    *,
    tenant_id: str,
    document_id: str,
    chunk_id: str,
    content: str,
) -> RetrievedChunk:
    return RetrievedChunk(
        tenant_id=tenant_id,
        workspace_id="hr",
        document_id=document_id,
        document_version=1,
        chunk_id=chunk_id,
        parent_id=f"parent-{document_id}",
        content=content,
        title="虚构制度",
        source_uri=f"memory://{document_id}",
    )


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"route": "shell", "confidence": 0.5, "reason": "bad"}, "route"),
        ({"route": "knowledge", "confidence": 1.1, "reason": "bad"}, "confidence"),
    ],
)
def test_route_decision_rejects_invalid_values(
    payload: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValidationError, match=match):
        RouteDecision.model_validate(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {"query": ""},
        {"query": "   "},
        {"query": "x" * 8001},
        {"query": "制度", "max_retries": 4},
        {"query": "制度", "tenant_id": "attacker-controlled"},
        {"query": "制度", "user_id": "attacker-controlled"},
        {"query": "制度", "roles": ["admin"]},
    ],
)
def test_query_request_rejects_invalid_or_identity_fields(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        QueryRequest.model_validate(payload)


def test_citation_requires_real_source_fields() -> None:
    with pytest.raises(ValidationError, match="chunk_id"):
        Citation.model_validate(
            {
                "document_id": "doc-1",
                "document_version": 1,
                "title": "制度",
                "source_uri": "memory://doc-1",
                "snippet": "证据",
            }
        )


def test_mutable_defaults_are_independent() -> None:
    first_principal = Principal(user_id="u1", tenant_id="t1")
    second_principal = Principal(user_id="u2", tenant_id="t2")
    first_principal.group_ids.append("finance")
    assert second_principal.group_ids == []

    first_grade = EvidenceGrade(
        relevant=False,
        sufficient=False,
        confidence=0,
        recommended_action="stop",
        reason="none",
    )
    second_grade = EvidenceGrade(
        relevant=False,
        sufficient=False,
        confidence=0,
        recommended_action="stop",
        reason="none",
    )
    first_grade.missing_aspects.append("policy")
    assert second_grade.missing_aspects == []

    first_response = QueryResponse(
        trace_id="trace-1",
        answer="",
        usage=Usage(),
        latency_ms=0,
        retries=0,
        finish_reason="insufficient_evidence",
    )
    second_response = QueryResponse(
        trace_id="trace-2",
        answer="",
        usage=Usage(),
        latency_ms=0,
        retries=0,
        finish_reason="insufficient_evidence",
    )
    first_response.citations.append(
        Citation(
            document_id="doc-1",
            document_version=1,
            chunk_id="chunk-1",
            title="制度",
            source_uri="memory://doc-1",
            snippet="证据",
        )
    )
    assert second_response.citations == []

    first_state = new_agent_state(trace_id="a", tenant_id="t", user_id="u")
    second_state = new_agent_state(trace_id="b", tenant_id="t", user_id="u")
    first_state["grade"]["reason"] = "changed"
    assert second_state["grade"] == {}


def test_principal_is_built_from_verified_identity() -> None:
    identity = VerifiedIdentity(
        user_id="user-1",
        tenant_id="tenant-a",
        group_ids=("hr",),
        scopes=("documents:read",),
    )
    principal = build_principal(identity)
    assert principal.model_dump() == {
        "user_id": "user-1",
        "tenant_id": "tenant-a",
        "group_ids": ["hr"],
        "scopes": ["documents:read"],
    }


@pytest.mark.asyncio
async def test_fake_embedding_is_deterministic_and_does_not_mutate_input() -> None:
    provider = FakeEmbedding(dimension=8)
    texts = ["年假 制度", "费用 报销"]
    before = deepcopy(texts)
    first = await provider.embed_documents(texts)
    second = await provider.embed_documents(texts)

    assert first == second
    assert texts == before
    assert len(first[0]) == 8
    assert isinstance(provider, EmbeddingProvider)


@pytest.mark.asyncio
async def test_fake_llm_validates_structured_output_without_network() -> None:
    response = '{"route":"knowledge","confidence":0.9,"reason":"internal policy"}'
    provider = FakeLLM({"route this": response})
    messages = [LLMMessage(role="user", content="route this")]

    result = await provider.complete(messages, RouteDecision, timeout_seconds=1)

    assert isinstance(result.parsed, RouteDecision)
    assert result.parsed.route == "knowledge"
    assert messages[0].content == "route this"
    assert len(provider.calls) == 1
    assert isinstance(provider, LLMProvider)


@pytest.mark.asyncio
async def test_fake_vector_store_prefilters_tenant_and_allowlist() -> None:
    embedder = FakeEmbedding()
    store = FakeVectorStore()
    allowed = make_chunk(
        tenant_id="tenant-a",
        document_id="doc-a",
        chunk_id="chunk-a",
        content="试用期员工年假制度",
    )
    forbidden = make_chunk(
        tenant_id="tenant-b",
        document_id="doc-b",
        chunk_id="chunk-b",
        content="试用期员工秘密薪酬制度",
    )
    embeddings = await embedder.embed_documents([allowed.content, forbidden.content])
    await store.upsert([allowed, forbidden], embeddings, "index-v1")

    query_vector = await embedder.embed_query("试用期 员工")
    auth_filter = AuthFilter(
        tenant_id="tenant-a",
        workspace_id="hr",
        group_ids=frozenset({"hr"}),
        allowed_document_ids=frozenset({"doc-a", "doc-b"}),
        allowed_index_versions=frozenset({"index-v1"}),
    )
    results = await store.search_dense(query_vector, auth_filter, top_k=5)

    assert [chunk.chunk_id for chunk in results] == ["chunk-a"]
    assert forbidden.score is None
    assert isinstance(store, VectorStore)


@pytest.mark.asyncio
async def test_fake_vector_store_fails_closed_without_allowed_documents() -> None:
    store = FakeVectorStore()
    chunk = make_chunk(
        tenant_id="tenant-a",
        document_id="doc-a",
        chunk_id="chunk-a",
        content="年假制度",
    )
    await store.upsert([chunk], [[1.0, 0.0]], "index-v1")
    auth_filter = AuthFilter(tenant_id="tenant-a", workspace_id="hr")

    assert await store.search_dense([1.0, 0.0], auth_filter, top_k=1) == []
    assert await store.search_lexical("年假", auth_filter, top_k=1) == []


def test_fake_container_satisfies_replaceable_protocols() -> None:
    container = build_fake_container()
    assert isinstance(container.llm, LLMProvider)
    assert isinstance(container.embedding, EmbeddingProvider)
    assert isinstance(container.vector_store, VectorStore)
