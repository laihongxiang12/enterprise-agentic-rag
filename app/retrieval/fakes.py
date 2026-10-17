"""Deterministic, network-free retrieval test adapters."""

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass

from app.schemas.retrieval import AuthFilter, RetrievedChunk


class FakeEmbedding:
    """Small stable hashing embedder suitable only for contract tests."""

    def __init__(self, dimension: int = 8) -> None:
        if dimension < 1:
            raise ValueError("dimension must be positive")
        self.dimension = dimension

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in text.casefold().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        return vector if norm == 0 else [value / norm for value in vector]


@dataclass(frozen=True)
class _StoredChunk:
    chunk: RetrievedChunk
    embedding: tuple[float, ...]
    index_version: str


class FakeVectorStore:
    """In-memory store that applies authorization before scoring candidates."""

    def __init__(self) -> None:
        self._records: dict[tuple[str, str, str, int, str], _StoredChunk] = {}

    async def upsert(
        self,
        chunks: Sequence[RetrievedChunk],
        embeddings: Sequence[Sequence[float]],
        version: str,
    ) -> list[str]:
        if not version:
            raise ValueError("version must not be empty")
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        inserted: list[str] = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            key = (
                chunk.tenant_id,
                chunk.workspace_id,
                chunk.document_id,
                chunk.document_version,
                chunk.chunk_id,
            )
            self._records[key] = _StoredChunk(
                chunk=chunk.model_copy(deep=True),
                embedding=tuple(float(value) for value in embedding),
                index_version=version,
            )
            inserted.append(chunk.chunk_id)
        return inserted

    async def search_dense(
        self,
        query_vector: Sequence[float],
        auth_filter: AuthFilter,
        top_k: int,
    ) -> list[RetrievedChunk]:
        self._validate_top_k(top_k)
        query = tuple(float(value) for value in query_vector)
        scored: list[tuple[float, RetrievedChunk]] = []
        for record in self._authorized_records(auth_filter):
            if len(query) != len(record.embedding):
                raise ValueError("query and stored embedding dimensions differ")
            score = self._cosine_similarity(query, record.embedding)
            scored.append((score, record.chunk))
        return self._ranked_copies(scored, top_k)

    async def search_lexical(
        self,
        query: str,
        auth_filter: AuthFilter,
        top_k: int,
    ) -> list[RetrievedChunk]:
        self._validate_top_k(top_k)
        query_terms = {char for char in query.casefold() if not char.isspace()}
        scored: list[tuple[float, RetrievedChunk]] = []
        for record in self._authorized_records(auth_filter):
            content_terms = {
                char for char in record.chunk.content.casefold() if not char.isspace()
            }
            score = (
                len(query_terms & content_terms) / len(query_terms)
                if query_terms
                else 0.0
            )
            if score > 0:
                scored.append((score, record.chunk))
        return self._ranked_copies(scored, top_k)

    async def delete_version(
        self,
        tenant_id: str,
        document_id: str,
        version: int,
    ) -> None:
        keys = [
            key
            for key in self._records
            if key[0] == tenant_id and key[2] == document_id and key[3] == version
        ]
        for key in keys:
            del self._records[key]

    def _authorized_records(self, auth_filter: AuthFilter) -> list[_StoredChunk]:
        if not auth_filter.allowed_document_ids:
            return []
        return [
            record
            for record in self._records.values()
            if record.chunk.tenant_id == auth_filter.tenant_id
            and record.chunk.workspace_id == auth_filter.workspace_id
            and record.chunk.document_id in auth_filter.allowed_document_ids
        ]

    @staticmethod
    def _validate_top_k(top_k: int) -> None:
        if top_k < 1:
            raise ValueError("top_k must be positive")

    @staticmethod
    def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return sum(a * b for a, b in zip(left, right, strict=True)) / (
            left_norm * right_norm
        )

    @staticmethod
    def _ranked_copies(
        scored: list[tuple[float, RetrievedChunk]],
        top_k: int,
    ) -> list[RetrievedChunk]:
        ranked = sorted(scored, key=lambda item: (-item[0], item[1].chunk_id))[:top_k]
        return [
            chunk.model_copy(update={"score": score}, deep=True)
            for score, chunk in ranked
        ]
