"""Stage 09 reranker, parent-context, and citation security tests."""

import asyncio
from collections.abc import Sequence
from unittest.mock import AsyncMock, Mock

import pytest

from app.ingestion.chunking import ParentChunk
from app.retrieval.base import Reranker
from app.retrieval.citations import CitationBuilder, CitationBuildError, CitationRequest
from app.retrieval.context import (
    ContextItem,
    ExpandedContext,
    ParentAuthorizationError,
    ParentContextExpander,
)
from app.retrieval.reranker import (
    FakeReranker,
    LocalCrossEncoderReranker,
    RerankerAuthorizationError,
    RerankService,
)
from app.schemas.retrieval import AuthFilter, RetrievedChunk


def chunk(
    number: int,
    content: str,
    *,
    tenant_id: str = "tenant-a",
    document_id: str | None = None,
    index_version: str = "index-active",
    parent_id: str | None = None,
    page: int | None = 2,
) -> RetrievedChunk:
    resolved_document = document_id or f"doc-{number}"
    return RetrievedChunk(
        tenant_id=tenant_id,
        workspace_id="hr",
        document_id=resolved_document,
        document_version=1,
        chunk_id=f"chunk-{number}",
        parent_id=parent_id or f"parent-{number}",
        content=content,
        title="员工制度",
        source_uri=f"fixture:///{tenant_id}/{resolved_document}",
        visibility="restricted",
        acl_groups=["hr"],
        index_version=index_version,
        page=page,
        score=1.0 / (number + 1),
    )


def scope(*chunks: RetrievedChunk) -> AuthFilter:
    return AuthFilter(
        tenant_id="tenant-a",
        workspace_id="hr",
        group_ids=frozenset({"hr"}),
        allowed_document_ids=frozenset(item.document_id for item in chunks),
        allowed_index_versions=frozenset(
            item.index_version for item in chunks if item.index_version is not None
        ),
        allowed_document_versions=frozenset(
            (item.document_id, item.document_version, item.index_version)
            for item in chunks
            if item.index_version is not None
        ),
    )


def parent(child: RetrievedChunk, content: str, **updates: object) -> ParentChunk:
    values: dict[str, object] = {
        "tenant_id": child.tenant_id,
        "workspace_id": child.workspace_id,
        "document_id": child.document_id,
        "document_version": child.document_version,
        "parent_id": child.parent_id,
        "content": content,
        "title": child.title,
        "source_uri": child.source_uri,
        "page": child.page,
        "heading_path": ["休假"],
        "content_hash": "hash",
        "acl_groups": child.acl_groups,
        "acl_version": 1,
        "index_version": child.index_version,
        "visibility": child.visibility,
    }
    values.update(updates)
    return ParentChunk.model_validate(values)


class ParentMemory:
    def __init__(self, parents: Sequence[ParentChunk]) -> None:
        self.parents = list(parents)
        self.calls: list[tuple[tuple[str, ...], AuthFilter]] = []

    async def get_parents(
        self, parent_ids: Sequence[str], auth_filter: AuthFilter
    ) -> list[ParentChunk]:
        self.calls.append((tuple(parent_ids), auth_filter))
        requested = set(parent_ids)
        return [item.model_copy(deep=True) for item in self.parents if item.parent_id in requested]


@pytest.mark.asyncio
async def test_fake_reranker_reorders_top20_to_top8_deterministically() -> None:
    candidates = [chunk(index, f"普通条款 {index}") for index in range(20)]
    candidates[19] = candidates[19].model_copy(update={"content": "年假申请特别条款"})
    service = RerankService(FakeReranker())

    outcome = await service.arerank("年假申请", candidates, scope(*candidates))

    assert isinstance(FakeReranker(), Reranker)
    assert len(outcome.chunks) == 8
    assert outcome.chunks[0].chunk_id == "chunk-19"
    assert not outcome.degraded


@pytest.mark.asyncio
async def test_local_cross_encoder_batches_pairs_and_validates_scores() -> None:
    candidates = [chunk(0, "A"), chunk(1, "B")]
    model = Mock()
    model.predict.return_value = [0.1, 0.9]
    adapter = LocalCrossEncoderReranker(model)

    results = await adapter.arerank("query", candidates, top_k=2)

    assert [item.chunk_id for item in results] == ["chunk-1", "chunk-0"]
    model.predict.assert_called_once_with([("query", "A"), ("query", "B")])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "expected_code"),
    [(RuntimeError("offline"), "dependency_error"), (None, "timeout")],
)
async def test_reranker_failure_has_bounded_original_order_fallback(
    failure: Exception | None, expected_code: str
) -> None:
    candidates = [chunk(index, f"内容 {index}") for index in range(12)]
    reranker = Mock()
    if failure is None:

        async def slow(*_args: object, **_kwargs: object) -> list[RetrievedChunk]:
            await asyncio.sleep(0.05)
            return []

        reranker.arerank = slow
    else:
        reranker.arerank = AsyncMock(side_effect=failure)
    service = RerankService(reranker, timeout_seconds=0.001)

    outcome = await service.arerank("年假", candidates, scope(*candidates))

    assert [item.chunk_id for item in outcome.chunks] == [f"chunk-{index}" for index in range(8)]
    assert outcome.degraded
    assert outcome.failure_code == expected_code


@pytest.mark.asyncio
async def test_unauthorized_candidate_never_reaches_reranker() -> None:
    allowed = chunk(0, "年假")
    forbidden = chunk(1, "秘密", tenant_id="tenant-b")
    reranker = Mock()
    reranker.arerank = AsyncMock()
    service = RerankService(reranker)

    with pytest.raises(RerankerAuthorizationError):
        await service.arerank("年假", [allowed, forbidden], scope(allowed))
    reranker.arerank.assert_not_called()


@pytest.mark.asyncio
async def test_invalid_empty_reranker_result_uses_original_order_fallback() -> None:
    candidates = [chunk(index, f"内容 {index}") for index in range(8)]
    reranker = Mock()
    reranker.arerank = AsyncMock(return_value=[])
    service = RerankService(reranker)

    outcome = await service.arerank("年假", candidates, scope(*candidates))

    assert outcome.degraded
    assert outcome.failure_code == "invalid_result"
    assert [item.chunk_id for item in outcome.chunks] == [f"chunk-{index}" for index in range(8)]


@pytest.mark.asyncio
async def test_parent_expansion_deduplicates_and_obeys_token_budget() -> None:
    first = chunk(0, "年假申请", parent_id="shared")
    second = chunk(1, "重复 child", document_id=first.document_id, parent_id="shared")
    third = chunk(2, "差旅申请", parent_id="other")
    store = ParentMemory(
        [
            parent(first, "员工年假申请需要提前三个工作日提交"),
            parent(third, "员工年假申请需要提前三个工作日提交"),
        ]
    )
    expander = ParentContextExpander(store, token_budget=6, max_parents=8)

    result = await expander.expand([first, second, third], scope(first, second, third))

    assert len(store.calls) == 1
    assert store.calls[0][0] == ("shared", "other")
    assert len(result.items) == 1
    assert result.items[0].used_parent
    assert result.items[0].truncated
    assert result.total_tokens == 6
    assert result.total_tokens <= result.token_budget


@pytest.mark.asyncio
async def test_missing_parent_safely_falls_back_to_authorized_child() -> None:
    child = chunk(0, "年假申请 child")
    result = await ParentContextExpander(ParentMemory([]), token_budget=20).expand(
        [child], scope(child)
    )
    assert result.items[0].content == child.content
    assert not result.items[0].used_parent


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "updates",
    [
        {"tenant_id": "tenant-b"},
        {"index_version": "index-stale"},
        {"acl_groups": ["finance"]},
        {"source_uri": "fixture:///forged"},
    ],
)
async def test_unauthorized_or_mismatched_parent_is_rejected(
    updates: dict[str, object],
) -> None:
    child = chunk(0, "年假申请")
    store = ParentMemory([parent(child, "完整父段落", **updates)])

    with pytest.raises(ParentAuthorizationError):
        await ParentContextExpander(store).expand([child], scope(child))


def test_citations_copy_real_child_source_page_and_bounded_snippet() -> None:
    child = chunk(0, "年假申请需要提前三个工作日提交", page=7)
    context = ExpandedContext(
        items=[
            ContextItem(
                child=child,
                content="更长的授权父段落",
                used_parent=True,
                truncated=False,
                token_count=9,
            )
        ],
        total_tokens=9,
        token_budget=20,
    )

    citations = CitationBuilder(snippet_max_chars=8).build(context, scope(child))

    assert citations[0].model_dump() == {
        "document_id": child.document_id,
        "document_version": 1,
        "chunk_id": child.chunk_id,
        "title": child.title,
        "source_uri": child.source_uri,
        "snippet": "年假申请需要提前",
        "page": 7,
    }


@pytest.mark.parametrize(
    "citation_request",
    [
        CitationRequest(document_id="missing", document_version=1, chunk_id="chunk-0"),
        CitationRequest(document_id="doc-0", document_version=2, chunk_id="chunk-0"),
        CitationRequest(document_id="doc-0", document_version=1, chunk_id="missing"),
    ],
)
def test_missing_or_version_mismatched_citation_fails(
    citation_request: CitationRequest,
) -> None:
    child = chunk(0, "年假申请")
    context = ExpandedContext(
        items=[
            ContextItem(
                child=child,
                content=child.content,
                used_parent=False,
                truncated=False,
                token_count=4,
            )
        ],
        total_tokens=4,
        token_budget=20,
    )

    with pytest.raises(CitationBuildError):
        CitationBuilder().build(context, scope(child), requests=[citation_request])


def test_unauthorized_evidence_cannot_become_a_citation() -> None:
    allowed = chunk(0, "年假申请")
    forged = chunk(0, "秘密", tenant_id="tenant-b")
    context = ExpandedContext(
        items=[
            ContextItem(
                child=forged,
                content=forged.content,
                used_parent=False,
                truncated=False,
                token_count=2,
            )
        ],
        total_tokens=2,
        token_budget=20,
    )
    with pytest.raises(CitationBuildError):
        CitationBuilder().build(context, scope(allowed))
