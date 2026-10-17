"""Offline hybrid retrieval with pre-scoring authorization and deterministic RRF."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from app.retrieval.base import EmbeddingProvider, VectorStore
from app.schemas.retrieval import AuthFilter, RetrievedChunk

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+|[\u3400-\u4dbf\u4e00-\u9fff]+", re.IGNORECASE)


def tokenize_text(text: str) -> list[str]:
    """Tokenize Latin words and CJK unigrams/bigrams without external models."""
    tokens: list[str] = []
    for match in _TOKEN_PATTERN.finditer(text.casefold()):
        value = match.group(0)
        if value.isascii():
            tokens.append(value)
            continue
        tokens.extend(value)
        tokens.extend(value[index : index + 2] for index in range(len(value) - 1))
    return tokens


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[RetrievedChunk]],
    *,
    top_k: int,
    k: int = 60,
) -> list[RetrievedChunk]:
    """Fuse rankings using one best-rank contribution per source and chunk."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if k < 1:
        raise ValueError("RRF k must be positive")

    scores: dict[tuple[str, str, str, int, str], float] = {}
    canonical: dict[tuple[str, str, str, int, str], tuple[int, int, RetrievedChunk]] = {}
    for source_priority, ranked in enumerate(ranked_lists):
        seen_in_source: set[tuple[str, str, str, int, str]] = set()
        for rank, chunk in enumerate(ranked, start=1):
            identity = _chunk_identity(chunk)
            if identity in seen_in_source:
                continue
            seen_in_source.add(identity)
            scores[identity] = scores.get(identity, 0.0) + 1.0 / (k + rank)
            candidate = (rank, source_priority, chunk)
            if identity not in canonical or candidate[:2] < canonical[identity][:2]:
                canonical[identity] = candidate

    ordered = sorted(
        scores,
        key=lambda identity: (
            -scores[identity],
            canonical[identity][0],
            canonical[identity][1],
            identity,
        ),
    )[:top_k]
    return [
        canonical[identity][2].model_copy(update={"score": scores[identity]}, deep=True)
        for identity in ordered
    ]


class OfflineHybridRetriever:
    """Dense + BM25 test adapter; it never performs an unfiltered search."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding: EmbeddingProvider,
        *,
        rrf_k: int = 60,
        candidate_multiplier: int = 3,
    ) -> None:
        if rrf_k < 1 or candidate_multiplier < 1:
            raise ValueError("rrf_k and candidate_multiplier must be positive")
        self._vector_store = vector_store
        self._embedding = embedding
        self._rrf_k = rrf_k
        self._candidate_multiplier = candidate_multiplier

    async def aretrieve(
        self,
        query: str,
        auth_filter: AuthFilter,
        top_k: int,
    ) -> list[RetrievedChunk]:
        if not query.strip():
            return []
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if not auth_filter.allowed_document_ids or not auth_filter.allowed_index_versions:
            return []

        query_vector = await self._embedding.embed_query(query)
        candidate_k = top_k * self._candidate_multiplier
        dense, lexical = await asyncio.gather(
            self._vector_store.search_dense(query_vector, auth_filter, top_k=candidate_k),
            self._vector_store.search_lexical(query, auth_filter, top_k=candidate_k),
        )
        return reciprocal_rank_fusion((dense, lexical), top_k=top_k, k=self._rrf_k)


@dataclass(frozen=True)
class HybridRetrievalRun:
    """One reproducible two-branch retrieval run, ready for eval export."""

    query: str
    dense_top_k: int
    lexical_top_k: int
    rrf_k: int
    fusion_top_k: int
    dense: tuple[RetrievedChunk, ...]
    lexical: tuple[RetrievedChunk, ...]
    fused: tuple[RetrievedChunk, ...]

    def to_eval_record(self) -> dict[str, Any]:
        """Return a JSON-serializable record without recomputing retrieval."""
        return {
            "schema_version": "retrieval-baseline-v1",
            "query": self.query,
            "parameters": {
                "dense_top_k": self.dense_top_k,
                "lexical_top_k": self.lexical_top_k,
                "rrf_k": self.rrf_k,
                "fusion_top_k": self.fusion_top_k,
            },
            "dense": _eval_hits(self.dense),
            "lexical": _eval_hits(self.lexical),
            "fused": _eval_hits(self.fused),
        }


class HybridRetriever:
    """Dense 30 + native BM25 30 -> deterministic RRF 20 production path."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding: EmbeddingProvider,
        *,
        dense_top_k: int = 30,
        lexical_top_k: int = 30,
        rrf_k: int = 60,
        fusion_top_k: int = 20,
    ) -> None:
        limits = (dense_top_k, lexical_top_k, rrf_k, fusion_top_k)
        if any(value < 1 for value in limits):
            raise ValueError("hybrid retrieval limits must be positive")
        self._vector_store = vector_store
        self._embedding = embedding
        self._dense_top_k = dense_top_k
        self._lexical_top_k = lexical_top_k
        self._rrf_k = rrf_k
        self._fusion_top_k = fusion_top_k

    async def aretrieve(
        self,
        query: str,
        auth_filter: AuthFilter,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        run = await self.aretrieve_with_baseline(query, auth_filter, top_k=top_k)
        return list(run.fused)

    async def aretrieve_with_baseline(
        self,
        query: str,
        auth_filter: AuthFilter,
        *,
        top_k: int | None = None,
    ) -> HybridRetrievalRun:
        fusion_top_k = self._fusion_top_k if top_k is None else top_k
        if fusion_top_k < 1:
            raise ValueError("top_k must be positive")
        if not query.strip() or not _has_complete_authorization(auth_filter):
            return HybridRetrievalRun(
                query=query,
                dense_top_k=self._dense_top_k,
                lexical_top_k=self._lexical_top_k,
                rrf_k=self._rrf_k,
                fusion_top_k=fusion_top_k,
                dense=(),
                lexical=(),
                fused=(),
            )

        query_vector = await self._embedding.embed_query(query)
        dense, lexical = await asyncio.gather(
            self._vector_store.search_dense(query_vector, auth_filter, top_k=self._dense_top_k),
            self._vector_store.search_lexical(query, auth_filter, top_k=self._lexical_top_k),
        )
        fused = reciprocal_rank_fusion((dense, lexical), top_k=fusion_top_k, k=self._rrf_k)
        return HybridRetrievalRun(
            query=query,
            dense_top_k=self._dense_top_k,
            lexical_top_k=self._lexical_top_k,
            rrf_k=self._rrf_k,
            fusion_top_k=fusion_top_k,
            dense=tuple(dense),
            lexical=tuple(lexical),
            fused=tuple(fused),
        )


def _chunk_identity(chunk: RetrievedChunk) -> tuple[str, str, str, int, str]:
    return (
        chunk.tenant_id,
        chunk.workspace_id,
        chunk.document_id,
        chunk.document_version,
        chunk.chunk_id,
    )


def _has_complete_authorization(auth_filter: AuthFilter) -> bool:
    return bool(
        auth_filter.allowed_document_ids
        and auth_filter.allowed_index_versions
        and auth_filter.allowed_document_versions
    )


def _eval_hits(chunks: Sequence[RetrievedChunk]) -> list[dict[str, Any]]:
    return [
        {
            "rank": rank,
            "tenant_id": chunk.tenant_id,
            "workspace_id": chunk.workspace_id,
            "document_id": chunk.document_id,
            "document_version": chunk.document_version,
            "index_version": chunk.index_version,
            "chunk_id": chunk.chunk_id,
            "title": chunk.title,
            "content": chunk.content,
            "score": chunk.score,
        }
        for rank, chunk in enumerate(chunks, start=1)
    ]
