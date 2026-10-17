"""Narrow asynchronous retrieval boundaries."""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from app.schemas.agent import EvidenceGrade
from app.schemas.retrieval import AuthFilter, RetrievedChunk


@runtime_checkable
class EmbeddingProvider(Protocol):
    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


@runtime_checkable
class VectorStore(Protocol):
    async def upsert(
        self,
        chunks: Sequence[RetrievedChunk],
        embeddings: Sequence[Sequence[float]],
        version: str,
    ) -> list[str]: ...

    async def search_dense(
        self,
        query_vector: Sequence[float],
        auth_filter: AuthFilter,
        top_k: int,
    ) -> list[RetrievedChunk]: ...

    async def search_lexical(
        self,
        query: str,
        auth_filter: AuthFilter,
        top_k: int,
    ) -> list[RetrievedChunk]: ...

    async def delete_version(
        self,
        tenant_id: str,
        workspace_id: str,
        document_id: str,
        document_version: int,
        index_version: str,
    ) -> None: ...


@runtime_checkable
class Retriever(Protocol):
    async def aretrieve(
        self,
        query: str,
        auth_filter: AuthFilter,
        top_k: int,
    ) -> list[RetrievedChunk]: ...


@runtime_checkable
class Reranker(Protocol):
    async def arerank(
        self,
        query: str,
        chunks: Sequence[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]: ...


@runtime_checkable
class Grader(Protocol):
    async def agrade(
        self,
        query: str,
        chunks: Sequence[RetrievedChunk],
    ) -> EvidenceGrade: ...
