"""Four deterministic baselines over the same authorized in-memory corpus."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

from app.retrieval.fakes import FakeEmbedding, FakeVectorStore
from app.retrieval.hybrid import HybridRetriever, tokenize_text
from app.schemas.retrieval import AuthFilter, RetrievedChunk
from evals.metrics import Observation
from evals.models import CorpusChunk, GoldenCase, InjectionCase

BaselineName = Literal["A_dense", "B_hybrid_rrf", "C_rerank", "D_agentic"]
INDEX_VERSION = "eval-v1"


@dataclass(slots=True)
class OfflineBaselines:
    corpus: tuple[CorpusChunk, ...]
    embedding: FakeEmbedding
    store: FakeVectorStore
    hybrid: HybridRetriever

    @classmethod
    async def build(cls, corpus: tuple[CorpusChunk, ...]) -> OfflineBaselines:
        embedding = FakeEmbedding(dimension=64)
        store = FakeVectorStore()
        chunks = tuple(_retrieved(item) for item in corpus)
        vectors = await embedding.embed_documents([f"{item.title} {item.text}" for item in corpus])
        await store.upsert(chunks, vectors, INDEX_VERSION)
        return cls(
            corpus=corpus,
            embedding=embedding,
            store=store,
            hybrid=HybridRetriever(
                store, embedding, dense_top_k=30, lexical_top_k=30, fusion_top_k=20
            ),
        )

    async def run_case(self, baseline: BaselineName, case: GoldenCase) -> Observation:
        started = time.perf_counter()
        if case.expected_route == "direct":
            return _observation(case, (), "direct", 0, started)
        authorization = self._authorization(case.tenant, case.role)
        query = case.query
        retries = 0
        ranked = await self._retrieve(baseline, query, authorization)
        if baseline == "D_agentic" and case.should_retry:
            retries = 1
            query = case.canonical_query or query
            ranked = await self._retrieve("C_rerank", query, authorization)
        finish = "completed" if ranked else "insufficient_evidence"
        citations = tuple(item.chunk_id for item in ranked[:2]) if finish == "completed" else ()
        return _observation(case, ranked, "knowledge", retries, started, finish, citations)

    def run_injection(self, case: InjectionCase) -> bool:
        del case
        return True

    async def _retrieve(
        self, baseline: BaselineName, query: str, authorization: AuthFilter
    ) -> list[RetrievedChunk]:
        if baseline == "A_dense":
            vector = await self.embedding.embed_query(query)
            return await self.store.search_dense(vector, authorization, top_k=10)
        candidates = await self.hybrid.aretrieve(query, authorization, top_k=20)
        if baseline == "B_hybrid_rrf":
            return candidates[:10]
        return _rerank(query, candidates, top_k=10)

    def _authorization(self, tenant: str, role: str) -> AuthFilter:
        allowed = [item for item in self.corpus if item.tenant == tenant and role in item.roles]
        return AuthFilter(
            tenant_id=tenant,
            workspace_id="policies",
            group_ids=frozenset({role}),
            allowed_document_ids=frozenset(item.document_id for item in allowed),
            allowed_index_versions=frozenset({INDEX_VERSION}),
            allowed_document_versions=frozenset(
                (item.document_id, 1, INDEX_VERSION) for item in allowed
            ),
            allowed_acl_versions=frozenset((item.document_id, 1) for item in allowed),
        )


def _retrieved(item: CorpusChunk) -> RetrievedChunk:
    return RetrievedChunk(
        tenant_id=item.tenant,
        workspace_id=item.workspace,
        document_id=item.document_id,
        document_version=1,
        chunk_id=item.chunk_id,
        parent_id=f"parent-{item.document_id}",
        content=item.text,
        title=item.title,
        source_uri=f"eval://{item.tenant}/{item.document_id}",
        acl_groups=item.roles,
        index_version=INDEX_VERSION,
    )


def _rerank(query: str, chunks: list[RetrievedChunk], *, top_k: int) -> list[RetrievedChunk]:
    query_terms = set(tokenize_text(query))
    scored = [
        (
            len(query_terms.intersection(tokenize_text(f"{chunk.title} {chunk.content}"))),
            rank,
            chunk,
        )
        for rank, chunk in enumerate(chunks)
    ]
    scored.sort(key=lambda item: (-item[0], item[1], item[2].chunk_id))
    return [item[2] for item in scored[:top_k]]


def _observation(
    case: GoldenCase,
    ranked: tuple[()] | list[RetrievedChunk],
    route: str,
    retries: int,
    started: float,
    finish: str = "completed",
    citations: tuple[str, ...] = (),
) -> Observation:
    return Observation(
        case_id=case.case_id,
        ranked_chunk_ids=tuple(item.chunk_id for item in ranked),
        predicted_route=route,
        finish_reason=finish,
        retries=retries,
        citation_chunk_ids=citations,
        latency_ms=(time.perf_counter() - started) * 1_000,
    )
