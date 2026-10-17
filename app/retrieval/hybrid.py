"""Offline hybrid retrieval with pre-scoring authorization and deterministic RRF."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence

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
    canonical: dict[
        tuple[str, str, str, int, str], tuple[int, int, RetrievedChunk]
    ] = {}
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
        canonical[identity][2].model_copy(
            update={"score": scores[identity]}, deep=True
        )
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
        if (
            not auth_filter.allowed_document_ids
            or not auth_filter.allowed_index_versions
        ):
            return []

        query_vector = await self._embedding.embed_query(query)
        candidate_k = top_k * self._candidate_multiplier
        dense, lexical = await asyncio.gather(
            self._vector_store.search_dense(
                query_vector, auth_filter, top_k=candidate_k
            ),
            self._vector_store.search_lexical(query, auth_filter, top_k=candidate_k),
        )
        return reciprocal_rank_fusion(
            (dense, lexical), top_k=top_k, k=self._rrf_k
        )


def _chunk_identity(chunk: RetrievedChunk) -> tuple[str, str, str, int, str]:
    return (
        chunk.tenant_id,
        chunk.workspace_id,
        chunk.document_id,
        chunk.document_version,
        chunk.chunk_id,
    )
