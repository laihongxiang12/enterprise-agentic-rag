"""Deterministic heading/page-aware parent-child chunking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal
from uuid import UUID, uuid5

from pydantic import Field

from app.ingestion.dedup import canonicalize_text
from app.ingestion.models import LoadedDocument
from app.schemas.base import ContractModel

NonEmptyString = Annotated[str, Field(min_length=1)]
DEFAULT_ID_NAMESPACE = UUID("d17417b6-3989-5d43-9f65-e210457031a1")
_BREAK_SEPARATORS = (
    "\n\n",
    "\n",
    "。",
    "\uff01",
    "\uff1f",
    ". ",
    "; ",
    "\uff0c",
    ", ",
    " ",
)


class ChunkContext(ContractModel):
    """Trusted metadata copied to every indexable chunk."""

    tenant_id: NonEmptyString
    workspace_id: NonEmptyString
    document_id: NonEmptyString
    document_version: int = Field(ge=1)
    content_hash: NonEmptyString
    acl_groups: list[str] = Field(default_factory=list)
    acl_version: int = Field(ge=1)
    index_version: NonEmptyString
    visibility: Literal["restricted", "tenant"] = "restricted"


class ParentChunk(ContractModel):
    """Larger generation context tied to one source section or page."""

    tenant_id: NonEmptyString
    workspace_id: NonEmptyString
    document_id: NonEmptyString
    document_version: int = Field(ge=1)
    parent_id: NonEmptyString
    content: NonEmptyString
    title: NonEmptyString
    source_uri: NonEmptyString
    page: int | None = Field(default=None, ge=1)
    heading_path: list[str] = Field(default_factory=list)
    content_hash: NonEmptyString
    acl_groups: list[str] = Field(default_factory=list)
    acl_version: int = Field(ge=1)
    index_version: NonEmptyString
    visibility: Literal["restricted", "tenant"] = "restricted"


class ChildChunk(ContractModel):
    """Smaller retrieval unit with complete source and authorization lineage."""

    tenant_id: NonEmptyString
    workspace_id: NonEmptyString
    document_id: NonEmptyString
    document_version: int = Field(ge=1)
    chunk_id: NonEmptyString
    parent_id: NonEmptyString
    content: NonEmptyString
    title: NonEmptyString
    source_uri: NonEmptyString
    page: int | None = Field(default=None, ge=1)
    heading_path: list[str] = Field(default_factory=list)
    content_hash: NonEmptyString
    acl_groups: list[str] = Field(default_factory=list)
    acl_version: int = Field(ge=1)
    index_version: NonEmptyString
    visibility: Literal["restricted", "tenant"] = "restricted"


class ChunkSet(ContractModel):
    """All parent and child chunks for one staged index version."""

    parents: list[ParentChunk] = Field(default_factory=list)
    children: list[ChildChunk] = Field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Character-based limits; overlap applies only to retrieval children."""

    parent_size: int = 1_200
    child_size: int = 400
    child_overlap: int = 60
    id_namespace: UUID = DEFAULT_ID_NAMESPACE

    def __post_init__(self) -> None:
        if self.parent_size < 1 or self.child_size < 1:
            raise ValueError("chunk sizes must be positive")
        if self.child_size > self.parent_size:
            raise ValueError("child_size cannot exceed parent_size")
        if not 0 <= self.child_overlap < self.child_size:
            raise ValueError("child_overlap must be non-negative and smaller than child_size")


class ParentChildChunker:
    """Split each source block into stable parent and child chunks."""

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()

    def chunk(self, document: LoadedDocument, context: ChunkContext) -> ChunkSet:
        parents: list[ParentChunk] = []
        children: list[ChildChunk] = []
        for block_index, block in enumerate(document.page_blocks):
            block_text = canonicalize_text(block.text)
            if not block_text:
                continue
            parent_parts = _split_text(block_text, self.config.parent_size, overlap=0)
            for parent_index, parent_text in enumerate(parent_parts):
                parent_id = self._stable_id(
                    "parent", context, block_index, parent_index, parent_text
                )
                parents.append(
                    ParentChunk(
                        **_common_fields(document, context),
                        parent_id=parent_id,
                        content=parent_text,
                        page=block.page,
                        heading_path=block.heading_path.copy(),
                    )
                )
                for child_index, child_text in enumerate(
                    _split_text(
                        parent_text,
                        self.config.child_size,
                        overlap=self.config.child_overlap,
                    )
                ):
                    children.append(
                        ChildChunk(
                            **_common_fields(document, context),
                            chunk_id=self._stable_id(
                                "child", context, parent_id, child_index, child_text
                            ),
                            parent_id=parent_id,
                            content=child_text,
                            page=block.page,
                            heading_path=block.heading_path.copy(),
                        )
                    )
        if not parents or not children:
            raise ValueError("document produced no chunks")
        return ChunkSet(parents=parents, children=children)

    def _stable_id(self, kind: str, context: ChunkContext, *parts: object) -> str:
        seed_parts = (
            kind,
            context.tenant_id,
            context.workspace_id,
            context.document_id,
            str(context.document_version),
            context.content_hash,
            *(str(part) for part in parts),
        )
        return str(uuid5(self.config.id_namespace, "\x1f".join(seed_parts)))


def _common_fields(document: LoadedDocument, context: ChunkContext) -> dict[str, object]:
    return {
        "tenant_id": context.tenant_id,
        "workspace_id": context.workspace_id,
        "document_id": context.document_id,
        "document_version": context.document_version,
        "title": document.title,
        "source_uri": document.source_uri,
        "content_hash": context.content_hash,
        "acl_groups": context.acl_groups.copy(),
        "acl_version": context.acl_version,
        "index_version": context.index_version,
        "visibility": context.visibility,
    }


def _split_text(text: str, max_size: int, *, overlap: int) -> list[str]:
    """Bounded splitter preferring paragraph and CJK punctuation boundaries."""
    if len(text) <= max_size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        hard_end = min(start + max_size, len(text))
        end = hard_end
        if hard_end < len(text):
            minimum_break = start + max(1, max_size // 2)
            for separator in _BREAK_SEPARATORS:
                position = text.rfind(separator, minimum_break, hard_end)
                if position >= minimum_break:
                    end = position + len(separator)
                    break
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks
