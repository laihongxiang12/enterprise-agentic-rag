"""Milvus 2.5 dense/BM25 adapter with mandatory server-side authorization."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from collections.abc import Sequence
from typing import Any
from uuid import uuid4

from pymilvus import (  # type: ignore[import-untyped]
    DataType,
    Function,
    FunctionType,
    MilvusClient,
)

from app.schemas.retrieval import AuthFilter, RetrievedChunk

_COLLECTION_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,254}$")
_OUTPUT_FIELDS = [
    "tenant_id",
    "workspace_id",
    "document_id",
    "document_version",
    "index_version",
    "chunk_id",
    "parent_id",
    "content",
    "title",
    "source_uri",
    "visibility",
    "acl_groups",
    "page",
    "heading_path",
]


class MilvusVectorStore:
    """Production-shaped dense and native BM25 adapter."""

    def __init__(
        self,
        uri: str,
        collection_name: str,
        dimension: int,
        *,
        timeout_seconds: float = 15.0,
    ) -> None:
        if not uri.strip():
            raise ValueError("Milvus URI must not be empty")
        if not _COLLECTION_NAME.fullmatch(collection_name):
            raise ValueError("invalid Milvus collection name")
        if dimension < 2:
            raise ValueError("embedding dimension must be at least 2")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._collection_name = collection_name
        self._dimension = dimension
        self._timeout = timeout_seconds
        self._uri = uri
        self._connection_alias = f"enterprise-rag-{uuid4().hex}"
        self._client: Any = None
        self._initialized = False
        self._initialize_lock = asyncio.Lock()

    async def initialize(self) -> None:
        if self._initialized:
            return
        async with self._initialize_lock:
            if self._initialized:
                return
            if self._client is None:
                self._client = await asyncio.to_thread(
                    MilvusClient,
                    uri=self._uri,
                    timeout=self._timeout,
                    alias=self._connection_alias,
                )
            exists = await asyncio.to_thread(
                self._client.has_collection,
                collection_name=self._collection_name,
                timeout=self._timeout,
            )
            if exists:
                await self._validate_existing_schema()
                await asyncio.to_thread(
                    self._client.load_collection,
                    collection_name=self._collection_name,
                    timeout=self._timeout,
                )
            else:
                await asyncio.to_thread(self._create_collection)
            self._initialized = True

    async def upsert(
        self,
        chunks: Sequence[RetrievedChunk],
        embeddings: Sequence[Sequence[float]],
        version: str,
    ) -> list[str]:
        if not version.strip():
            raise ValueError("index version must not be empty")
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            return []
        rows = [
            self._row(chunk, embedding, version)
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        await self.initialize()
        await asyncio.to_thread(
            self._client.upsert,
            collection_name=self._collection_name,
            data=rows,
            timeout=self._timeout,
        )
        await asyncio.to_thread(
            self._client.flush,
            collection_name=self._collection_name,
            timeout=self._timeout,
        )
        return [chunk.chunk_id for chunk in chunks]

    async def search_dense(
        self,
        query_vector: Sequence[float],
        auth_filter: AuthFilter,
        top_k: int,
    ) -> list[RetrievedChunk]:
        if top_k < 1 or top_k > 16_384:
            raise ValueError("top_k must be between 1 and 16384")
        vector = [float(value) for value in query_vector]
        self._validate_dimension(vector)
        if (
            not auth_filter.allowed_document_ids
            or not auth_filter.allowed_index_versions
            or not auth_filter.allowed_document_versions
        ):
            return []
        expression, parameters = _authorization_filter(auth_filter)
        await self.initialize()
        raw = await asyncio.to_thread(
            self._client.search,
            collection_name=self._collection_name,
            data=[vector],
            anns_field="dense_embedding",
            filter=expression,
            filter_params=parameters,
            limit=top_k,
            output_fields=_OUTPUT_FIELDS,
            search_params={"metric_type": "COSINE", "params": {}},
            consistency_level="Strong",
            timeout=self._timeout,
        )
        hits = raw[0] if raw else []
        return [_retrieved_chunk(hit) for hit in hits]

    async def search_lexical(
        self,
        query: str,
        auth_filter: AuthFilter,
        top_k: int,
    ) -> list[RetrievedChunk]:
        if top_k < 1 or top_k > 16_384:
            raise ValueError("top_k must be between 1 and 16384")
        if not query.strip():
            return []
        if (
            not auth_filter.allowed_document_ids
            or not auth_filter.allowed_index_versions
            or not auth_filter.allowed_document_versions
        ):
            return []
        expression, parameters = _authorization_filter(auth_filter)
        await self.initialize()
        raw = await asyncio.to_thread(
            self._client.search,
            collection_name=self._collection_name,
            data=[query],
            anns_field="sparse_embedding",
            filter=expression,
            filter_params=parameters,
            limit=top_k,
            output_fields=_OUTPUT_FIELDS,
            search_params={"metric_type": "BM25", "params": {}},
            consistency_level="Strong",
            timeout=self._timeout,
        )
        hits = raw[0] if raw else []
        return [_retrieved_chunk(hit) for hit in hits]

    async def delete_version(
        self,
        tenant_id: str,
        workspace_id: str,
        document_id: str,
        document_version: int,
        index_version: str,
    ) -> None:
        if document_version < 1:
            raise ValueError("document_version must be positive")
        if not all(
            value.strip() for value in (tenant_id, workspace_id, document_id, index_version)
        ):
            raise ValueError("delete scope values must not be empty")
        await self.initialize()
        await asyncio.to_thread(
            self._client.delete,
            collection_name=self._collection_name,
            filter=(
                "tenant_id == {tenant_id} AND workspace_id == {workspace_id} "
                "AND document_id == {document_id} "
                "AND document_version == {document_version} "
                "AND index_version == {index_version}"
            ),
            filter_params={
                "tenant_id": tenant_id,
                "workspace_id": workspace_id,
                "document_id": document_id,
                "document_version": document_version,
                "index_version": index_version,
            },
            timeout=self._timeout,
        )
        await asyncio.to_thread(
            self._client.flush,
            collection_name=self._collection_name,
            timeout=self._timeout,
        )

    async def drop_collection(self) -> None:
        """Drop this adapter's collection; intended for isolated test cleanup."""
        if self._client is None:
            return
        exists = await asyncio.to_thread(
            self._client.has_collection,
            collection_name=self._collection_name,
            timeout=self._timeout,
        )
        if exists:
            await asyncio.to_thread(
                self._client.drop_collection,
                collection_name=self._collection_name,
                timeout=self._timeout,
            )
        self._initialized = False

    async def close(self) -> None:
        if self._client is not None:
            await asyncio.to_thread(self._client.close)
            self._client = None
        self._initialized = False

    def _create_collection(self) -> None:
        schema = MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
            description="Authorized enterprise RAG child chunks",
        )
        schema.add_field(
            field_name="pk",
            datatype=DataType.VARCHAR,
            max_length=64,
            is_primary=True,
        )
        schema.add_field(
            field_name="dense_embedding",
            datatype=DataType.FLOAT_VECTOR,
            dim=self._dimension,
        )
        schema.add_field(
            field_name="lexical_text",
            datatype=DataType.VARCHAR,
            max_length=65_535,
            enable_analyzer=True,
            analyzer_params={"tokenizer": "jieba"},
        )
        schema.add_field(
            field_name="sparse_embedding",
            datatype=DataType.SPARSE_FLOAT_VECTOR,
        )
        schema.add_function(
            Function(
                name="lexical_bm25",
                function_type=FunctionType.BM25,
                input_field_names=["lexical_text"],
                output_field_names=["sparse_embedding"],
            )
        )
        _add_varchar(schema, "tenant_id", 128)
        _add_varchar(schema, "workspace_id", 128)
        _add_varchar(schema, "document_id", 256)
        schema.add_field(field_name="document_version", datatype=DataType.INT64)
        _add_varchar(schema, "index_version", 256)
        _add_varchar(schema, "chunk_id", 256)
        _add_varchar(schema, "parent_id", 256)
        _add_varchar(schema, "content", 65_535)
        _add_varchar(schema, "title", 1_024)
        _add_varchar(schema, "source_uri", 2_048)
        _add_varchar(schema, "visibility", 16)
        schema.add_field(
            field_name="acl_groups",
            datatype=DataType.ARRAY,
            element_type=DataType.VARCHAR,
            max_capacity=128,
            max_length=256,
        )
        schema.add_field(field_name="page", datatype=DataType.INT64, nullable=True)
        schema.add_field(
            field_name="heading_path",
            datatype=DataType.ARRAY,
            element_type=DataType.VARCHAR,
            max_capacity=64,
            max_length=1_024,
        )
        schema.add_field(field_name="ready", datatype=DataType.BOOL)

        indexes = self._client.prepare_index_params()
        indexes.add_index(
            field_name="dense_embedding",
            index_name="dense_embedding_idx",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        indexes.add_index(
            field_name="sparse_embedding",
            index_name="sparse_embedding_idx",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
            params={"inverted_index_algo": "DAAT_MAXSCORE"},
        )
        for field_name in (
            "tenant_id",
            "workspace_id",
            "document_id",
            "index_version",
            "visibility",
            "ready",
        ):
            indexes.add_index(
                field_name=field_name,
                index_name=f"{field_name}_idx",
                index_type="INVERTED",
            )
        self._client.create_collection(
            collection_name=self._collection_name,
            schema=schema,
            index_params=indexes,
            consistency_level="Strong",
            timeout=self._timeout,
        )

    async def _validate_existing_schema(self) -> None:
        description = await asyncio.to_thread(
            self._client.describe_collection,
            collection_name=self._collection_name,
            timeout=self._timeout,
        )
        fields = {field["name"]: field for field in description.get("fields", [])}
        if description.get("enable_dynamic_field", False):
            raise ValueError("Milvus collection must disable dynamic fields")
        required = {
            "pk",
            "dense_embedding",
            "lexical_text",
            "sparse_embedding",
            *_OUTPUT_FIELDS,
            "ready",
        }
        missing = required.difference(fields)
        if missing:
            raise ValueError(f"Milvus collection schema is missing fields: {sorted(missing)}")
        if fields["sparse_embedding"].get("type") != DataType.SPARSE_FLOAT_VECTOR:
            raise ValueError("Milvus sparse_embedding field must be SPARSE_FLOAT_VECTOR")
        lexical_parameters = fields["lexical_text"].get("params", {})
        analyzer_enabled = str(lexical_parameters.get("enable_analyzer", "")).lower()
        try:
            analyzer = json.loads(lexical_parameters.get("analyzer_params", "{}"))
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("Milvus lexical analyzer configuration is invalid") from error
        if analyzer_enabled != "true" or analyzer.get("tokenizer") != "jieba":
            raise ValueError("Milvus lexical_text field must use the Jieba analyzer")
        functions = description.get("functions", [])
        if not any(
            function.get("type") == FunctionType.BM25
            and function.get("input_field_names") == ["lexical_text"]
            and function.get("output_field_names") == ["sparse_embedding"]
            for function in functions
        ):
            raise ValueError("Milvus collection is missing the lexical BM25 function")
        indexes = set(
            await asyncio.to_thread(
                self._client.list_indexes,
                collection_name=self._collection_name,
                timeout=self._timeout,
            )
        )
        missing_indexes = {"dense_embedding_idx", "sparse_embedding_idx"}.difference(indexes)
        if missing_indexes:
            raise ValueError(f"Milvus collection is missing indexes: {sorted(missing_indexes)}")
        parameters = fields["dense_embedding"].get("params", {})
        actual_dimension = int(parameters.get("dim", 0))
        if actual_dimension != self._dimension:
            raise ValueError(
                "Milvus collection dimension does not match configured embedding dimension"
            )

    def _row(
        self,
        chunk: RetrievedChunk,
        embedding: Sequence[float],
        index_version: str,
    ) -> dict[str, object]:
        vector = [float(value) for value in embedding]
        self._validate_dimension(vector)
        acl_groups = sorted({group.strip() for group in chunk.acl_groups if group.strip()})
        if chunk.visibility == "restricted" and not acl_groups:
            raise ValueError("restricted chunks require at least one ACL group")
        if len(acl_groups) > 128 or any(len(group.encode()) > 256 for group in acl_groups):
            raise ValueError("ACL group limits exceed Milvus schema")
        if len(chunk.heading_path) > 64 or any(
            len(heading.encode()) > 1_024 for heading in chunk.heading_path
        ):
            raise ValueError("heading path limits exceed Milvus schema")
        values = (
            chunk.tenant_id,
            chunk.workspace_id,
            chunk.document_id,
            str(chunk.document_version),
            index_version,
            chunk.chunk_id,
        )
        primary_key = hashlib.sha256(json.dumps(values, ensure_ascii=False).encode()).hexdigest()
        return {
            "pk": primary_key,
            "dense_embedding": vector,
            "lexical_text": f"{chunk.title}\n{chunk.content}",
            "tenant_id": chunk.tenant_id,
            "workspace_id": chunk.workspace_id,
            "document_id": chunk.document_id,
            "document_version": chunk.document_version,
            "index_version": index_version,
            "chunk_id": chunk.chunk_id,
            "parent_id": chunk.parent_id,
            "content": chunk.content,
            "title": chunk.title,
            "source_uri": chunk.source_uri,
            "visibility": chunk.visibility,
            "acl_groups": acl_groups,
            "page": chunk.page,
            "heading_path": list(chunk.heading_path),
            "ready": True,
        }

    def _validate_dimension(self, vector: Sequence[float]) -> None:
        if len(vector) != self._dimension:
            raise ValueError(f"embedding dimension must be {self._dimension}, got {len(vector)}")
        if not all(math.isfinite(value) for value in vector):
            raise ValueError("embedding values must be finite")


def _add_varchar(schema: Any, field_name: str, max_length: int) -> None:
    schema.add_field(
        field_name=field_name,
        datatype=DataType.VARCHAR,
        max_length=max_length,
    )


def _authorization_filter(auth_filter: AuthFilter) -> tuple[str, dict[str, object]]:
    expression = (
        "tenant_id == {tenant_id} AND workspace_id == {workspace_id} "
        "AND document_id IN {document_ids} "
        "AND index_version IN {index_versions} AND ready == true AND "
    )
    parameters: dict[str, object] = {
        "tenant_id": auth_filter.tenant_id,
        "workspace_id": auth_filter.workspace_id,
        "document_ids": sorted(auth_filter.allowed_document_ids),
        "index_versions": sorted(auth_filter.allowed_index_versions),
        "tenant_visibility": "tenant",
    }
    if auth_filter.group_ids:
        expression += (
            "(visibility == {tenant_visibility} OR ARRAY_CONTAINS_ANY(acl_groups, {group_ids}))"
        )
        parameters["group_ids"] = sorted(auth_filter.group_ids)
    else:
        expression += "visibility == {tenant_visibility}"
    bindings = []
    for position, (document_id, document_version, index_version) in enumerate(
        sorted(auth_filter.allowed_document_versions)
    ):
        bindings.append(
            f"(document_id == {{doc_{position}}} "
            f"AND document_version == {{ver_{position}}} "
            f"AND index_version == {{idx_{position}}})"
        )
        parameters[f"doc_{position}"] = document_id
        parameters[f"ver_{position}"] = document_version
        parameters[f"idx_{position}"] = index_version
    expression += " AND (" + " OR ".join(bindings) + ")"
    return expression, parameters


def _retrieved_chunk(hit: dict[str, Any]) -> RetrievedChunk:
    entity = hit.get("entity", {})
    return RetrievedChunk.model_validate(
        {
            **entity,
            "score": float(hit["distance"]),
        }
    )
