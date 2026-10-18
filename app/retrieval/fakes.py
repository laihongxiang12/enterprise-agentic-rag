"""Deterministic, network-free retrieval test adapters."""

import hashlib
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from app.retrieval.hybrid import tokenize_text
from app.schemas.retrieval import AuthFilter, RetrievedChunk


class FakeRetriever:
    """Return authorized scripted chunks and record the exact trusted filter."""

    def __init__(
        self,
        responses: Mapping[str, Sequence[RetrievedChunk]] | None = None,
        *,
        failures: Mapping[str, Exception] | None = None,
    ) -> None:
        self._responses = {
            query: tuple(chunk.model_copy(deep=True) for chunk in chunks)
            for query, chunks in (responses or {}).items()
        }
        self._failures = dict(failures or {})
        self.calls: list[tuple[str, AuthFilter, int]] = []

    async def aretrieve(
        self,
        query: str,
        auth_filter: AuthFilter,
        top_k: int,
    ) -> list[RetrievedChunk]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self.calls.append((query, auth_filter.model_copy(deep=True), top_k))
        failure = self._failures.get(query)
        if failure is not None:
            raise failure
        return [chunk.model_copy(deep=True) for chunk in self._responses.get(query, ())][
            :top_k
        ]


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
        for token in tokenize_text(text):
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
        self._records: dict[tuple[str, str, str, int, str, str], _StoredChunk] = {}

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
                version,
                chunk.chunk_id,
            )
            self._records[key] = _StoredChunk(
                chunk=chunk.model_copy(update={"index_version": version}, deep=True),
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
            if score > 0:
                scored.append((score, record.chunk))
        return self._ranked_copies(scored, top_k)

    async def search_lexical(
        self,
        query: str,
        auth_filter: AuthFilter,
        top_k: int,
    ) -> list[RetrievedChunk]:
        self._validate_top_k(top_k)
        query_terms = tokenize_text(query)
        if not query_terms:
            return []
        records = self._authorized_records(auth_filter)
        tokenized = [
            tokenize_text(f"{record.chunk.title} {record.chunk.content}")
            for record in records
        ]
        document_frequency = Counter(
            token for document in tokenized for token in set(document)
        )
        average_length = (
            sum(len(document) for document in tokenized) / len(tokenized)
            if tokenized
            else 0.0
        )
        scored: list[tuple[float, RetrievedChunk]] = []
        for record, document_terms in zip(records, tokenized, strict=True):
            score = self._bm25_score(
                query_terms,
                document_terms,
                document_frequency,
                len(tokenized),
                average_length,
            )
            if score > 0:
                scored.append((score, record.chunk))
        return self._ranked_copies(scored, top_k)

    async def delete_version(
        self,
        tenant_id: str,
        workspace_id: str,
        document_id: str,
        document_version: int,
        index_version: str,
    ) -> None:
        keys = [
            key
            for key in self._records
            if key[0] == tenant_id
            and key[1] == workspace_id
            and key[2] == document_id
            and key[3] == document_version
            and key[4] == index_version
        ]
        for key in keys:
            del self._records[key]

    def _authorized_records(self, auth_filter: AuthFilter) -> list[_StoredChunk]:
        if (
            not auth_filter.allowed_document_ids
            or not auth_filter.allowed_index_versions
        ):
            return []
        return [
            record
            for record in self._records.values()
            if record.chunk.tenant_id == auth_filter.tenant_id
            and record.chunk.workspace_id == auth_filter.workspace_id
            and record.chunk.document_id in auth_filter.allowed_document_ids
            and record.index_version in auth_filter.allowed_index_versions
            and (
                not auth_filter.allowed_document_versions
                or (
                    record.chunk.document_id,
                    record.chunk.document_version,
                    record.index_version,
                ) in auth_filter.allowed_document_versions
            )
        ]

    @staticmethod
    def _bm25_score(
        query_terms: Sequence[str],
        document_terms: Sequence[str],
        document_frequency: Counter[str],
        corpus_size: int,
        average_length: float,
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> float:
        if corpus_size == 0 or average_length == 0:
            return 0.0
        frequencies = Counter(document_terms)
        length = len(document_terms)
        score = 0.0
        for term in set(query_terms):
            frequency = frequencies[term]
            if frequency == 0:
                continue
            documents_with_term = document_frequency[term]
            inverse_document_frequency = math.log(
                1.0
                + (corpus_size - documents_with_term + 0.5)
                / (documents_with_term + 0.5)
            )
            denominator = frequency + k1 * (
                1.0 - b + b * length / average_length
            )
            score += inverse_document_frequency * frequency * (k1 + 1.0) / denominator
        return score

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
