"""Transactional ingestion orchestration over replaceable stores."""

from __future__ import annotations

import hashlib
from typing import Annotated, Literal, Protocol, cast, runtime_checkable

from pydantic import Field, model_validator

from app.ingestion.chunking import ChunkContext, ChunkSet, ParentChildChunker
from app.ingestion.dedup import canonicalize_text, exact_content_hash, normalize_acl_groups
from app.ingestion.models import LoadedDocument, PageBlock
from app.schemas.base import ContractModel
from app.schemas.documents import DocumentMeta

NonEmptyString = Annotated[str, Field(min_length=1)]
SourceType = Literal["pdf", "markdown", "html"]
IngestionOutcome = Literal["indexed", "deduplicated", "acl_updated"]


class IngestionRequest(ContractModel):
    """Server-created ingestion scope and ACL input."""

    tenant_id: NonEmptyString
    workspace_id: NonEmptyString
    document_id: NonEmptyString
    acl_groups: list[str] = Field(default_factory=list)
    visibility: Literal["restricted", "tenant"] = "restricted"
    index_schema_version: NonEmptyString = "chunks-v1"

    @model_validator(mode="after")
    def validate_visibility_and_acl(self) -> IngestionRequest:
        """Permit empty ACLs only for explicitly tenant-visible documents."""
        self.acl_groups = list(normalize_acl_groups(self.acl_groups))
        if self.visibility == "restricted" and not self.acl_groups:
            raise ValueError("restricted documents require at least one ACL group")
        return self


class StoredVersion(ContractModel):
    """A staged or active metadata and chunk snapshot."""

    document: DocumentMeta
    chunks: ChunkSet


class IngestionResult(ContractModel):
    """Observable result without exposing mutable store internals."""

    outcome: IngestionOutcome
    document: DocumentMeta
    parent_count: int = Field(ge=0)
    child_count: int = Field(ge=0)


class IngestionFailure(RuntimeError):
    """A new version failed before activation; the old version remains active."""


@runtime_checkable
class IngestionMetadataStore(Protocol):
    """Metadata boundary whose activate operation must be atomic."""

    async def get_active(
        self, tenant_id: str, workspace_id: str, document_id: str
    ) -> StoredVersion | None: ...

    async def stage(self, version: StoredVersion) -> None: ...

    async def activate(
        self, tenant_id: str, workspace_id: str, document_id: str, index_version: str
    ) -> StoredVersion: ...

    async def mark_failed(self, index_version: str) -> None: ...


@runtime_checkable
class IndexWriter(Protocol):
    """Replaceable index boundary; validation occurs before metadata activation."""

    async def write_version(self, version: StoredVersion) -> None: ...

    async def validate_version(self, index_version: str) -> None: ...

    async def discard_version(self, index_version: str) -> None: ...


class IngestionPipeline:
    """Deduplicate, stage, validate, and atomically activate one document version."""

    def __init__(
        self,
        metadata_store: IngestionMetadataStore,
        index_writer: IndexWriter,
        chunker: ParentChildChunker | None = None,
    ) -> None:
        self._metadata_store = metadata_store
        self._index_writer = index_writer
        self._chunker = chunker or ParentChildChunker()

    async def ingest(
        self, loaded: LoadedDocument, request: IngestionRequest
    ) -> IngestionResult:
        canonical = _canonical_document(loaded)
        content_hash = exact_content_hash(canonical.text)
        acl_groups = normalize_acl_groups(request.acl_groups)
        active = await self._metadata_store.get_active(
            request.tenant_id, request.workspace_id, request.document_id
        )
        if (
            active is not None
            and active.document.content_hash == content_hash
            and tuple(active.document.acl_groups) == acl_groups
        ):
            return _result("deduplicated", active)

        content_changed = active is None or active.document.content_hash != content_hash
        acl_changed = active is None or tuple(active.document.acl_groups) != acl_groups
        document_version = 1 if active is None else active.document.version + int(content_changed)
        acl_version = 1 if active is None else active.document.acl_version + int(acl_changed)
        index_version = _index_version(
            request=request,
            document_version=document_version,
            acl_version=acl_version,
            content_hash=content_hash,
        )
        document = DocumentMeta(
            tenant_id=request.tenant_id,
            workspace_id=request.workspace_id,
            document_id=request.document_id,
            version=document_version,
            title=canonical.title,
            source_uri=canonical.source_uri,
            source_type=_source_type(canonical),
            visibility=request.visibility,
            acl_groups=list(acl_groups),
            content_hash=content_hash,
            acl_version=acl_version,
            index_version=index_version,
            status="processing",
        )
        chunks = self._chunker.chunk(
            canonical,
            ChunkContext(
                tenant_id=request.tenant_id,
                workspace_id=request.workspace_id,
                document_id=request.document_id,
                document_version=document_version,
                content_hash=content_hash,
                acl_groups=list(acl_groups),
                acl_version=acl_version,
                index_version=index_version,
                visibility=request.visibility,
            ),
        )
        staged = StoredVersion(document=document, chunks=chunks)
        await self._metadata_store.stage(staged)
        try:
            await self._index_writer.write_version(staged)
            await self._index_writer.validate_version(index_version)
            activated = await self._metadata_store.activate(
                request.tenant_id,
                request.workspace_id,
                request.document_id,
                index_version,
            )
        except Exception as exc:
            await self._index_writer.discard_version(index_version)
            await self._metadata_store.mark_failed(index_version)
            raise IngestionFailure("Index version failed before activation") from exc

        outcome: IngestionOutcome = "acl_updated" if not content_changed else "indexed"
        return _result(outcome, activated)


def _canonical_document(document: LoadedDocument) -> LoadedDocument:
    page_blocks = []
    for block in document.page_blocks:
        text = canonicalize_text(block.text)
        if text:
            page_blocks.append(
                PageBlock(
                    text=text,
                    page=block.page,
                    heading_path=block.heading_path.copy(),
                )
            )
    return document.model_copy(
        update={"text": canonicalize_text(document.text), "page_blocks": page_blocks},
        deep=True,
    )


def _source_type(document: LoadedDocument) -> SourceType:
    value = document.metadata.get("source_type")
    if value not in {"pdf", "markdown", "html"}:
        raise IngestionFailure("Loaded document has an invalid source_type")
    return cast(SourceType, value)


def _index_version(
    *,
    request: IngestionRequest,
    document_version: int,
    acl_version: int,
    content_hash: str,
) -> str:
    seed = "\x1f".join(
        (
            request.tenant_id,
            request.workspace_id,
            request.document_id,
            str(document_version),
            str(acl_version),
            request.index_schema_version,
            content_hash,
        )
    )
    return f"{request.index_schema_version}-{hashlib.sha256(seed.encode()).hexdigest()[:24]}"


def _result(outcome: IngestionOutcome, version: StoredVersion) -> IngestionResult:
    return IngestionResult(
        outcome=outcome,
        document=version.document.model_copy(deep=True),
        parent_count=len(version.chunks.parents),
        child_count=len(version.chunks.children),
    )
