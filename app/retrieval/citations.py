"""Citation construction from already-authorized retrieval evidence only."""

from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import Field

from app.retrieval.context import ContextItem, ExpandedContext
from app.schemas.api import Citation
from app.schemas.base import ContractModel
from app.schemas.retrieval import AuthFilter, RetrievedChunk

_SPACE = re.compile(r"\s+")


class CitationBuildError(ValueError):
    """A requested citation cannot be proven by the supplied evidence."""


class CitationRequest(ContractModel):
    """Internal reference emitted by later generation, never a source record."""

    document_id: str = Field(min_length=1)
    document_version: int = Field(ge=1)
    chunk_id: str = Field(min_length=1)


class CitationBuilder:
    """Resolve references against authorized evidence and copy source fields."""

    def __init__(self, *, snippet_max_chars: int = 320) -> None:
        if snippet_max_chars < 1:
            raise ValueError("snippet_max_chars must be positive")
        self._snippet_max_chars = snippet_max_chars

    def build(
        self,
        context: ExpandedContext,
        auth_filter: AuthFilter,
        requests: Sequence[CitationRequest] | None = None,
    ) -> list[Citation]:
        evidence = _index_evidence(context.items, auth_filter)
        selected = (
            list(requests)
            if requests is not None
            else [
                CitationRequest(
                    document_id=item.child.document_id,
                    document_version=item.child.document_version,
                    chunk_id=item.child.chunk_id,
                )
                for item in context.items
            ]
        )
        citations: list[Citation] = []
        seen: set[tuple[str, int, str]] = set()
        for request in selected:
            key = (request.document_id, request.document_version, request.chunk_id)
            if key in seen:
                continue
            item = evidence.get(key)
            if item is None:
                raise CitationBuildError("citation reference is missing or version-mismatched")
            child = item.child
            snippet = _SPACE.sub(" ", child.content).strip()[: self._snippet_max_chars].rstrip()
            if not snippet:
                raise CitationBuildError("citation evidence has no usable snippet")
            citations.append(
                Citation(
                    document_id=child.document_id,
                    document_version=child.document_version,
                    chunk_id=child.chunk_id,
                    title=child.title,
                    source_uri=child.source_uri,
                    snippet=snippet,
                    page=child.page,
                )
            )
            seen.add(key)
        return citations


def _index_evidence(
    items: Sequence[ContextItem], auth_filter: AuthFilter
) -> dict[tuple[str, int, str], ContextItem]:
    indexed: dict[tuple[str, int, str], ContextItem] = {}
    for item in items:
        child = item.child
        _validate_authorized(child, auth_filter)
        key = (child.document_id, child.document_version, child.chunk_id)
        existing = indexed.get(key)
        if existing is not None and existing.child != child:
            raise CitationBuildError("citation identity has conflicting source metadata")
        indexed[key] = item
    return indexed


def _validate_authorized(chunk: RetrievedChunk, auth_filter: AuthFilter) -> None:
    binding = (chunk.document_id, chunk.document_version, chunk.index_version)
    visible = chunk.visibility == "tenant" or bool(
        set(chunk.acl_groups).intersection(auth_filter.group_ids)
    )
    if (
        chunk.tenant_id != auth_filter.tenant_id
        or chunk.workspace_id != auth_filter.workspace_id
        or chunk.document_id not in auth_filter.allowed_document_ids
        or chunk.index_version not in auth_filter.allowed_index_versions
        or binding not in auth_filter.allowed_document_versions
        or not visible
    ):
        raise CitationBuildError("citation evidence is outside authorization scope")
