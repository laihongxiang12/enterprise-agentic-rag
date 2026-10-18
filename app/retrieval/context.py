"""Authorized parent expansion with deterministic context budgeting."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from pydantic import Field

from app.ingestion.chunking import ParentChunk
from app.schemas.base import ContractModel
from app.schemas.retrieval import AuthFilter, RetrievedChunk

_TOKEN = re.compile(r"[a-z0-9]+|[\u3400-\u4dbf\u4e00-\u9fff]|[^\s]", re.IGNORECASE)
_SPACE = re.compile(r"\s+")


class ParentAuthorizationError(ValueError):
    """A parent-store response violates the requested authorization scope."""


@runtime_checkable
class ParentStore(Protocol):
    """Storage boundary that must apply the supplied authorization pre-filter."""

    async def get_parents(
        self,
        parent_ids: Sequence[str],
        auth_filter: AuthFilter,
    ) -> list[ParentChunk]: ...


@runtime_checkable
class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...

    def truncate(self, text: str, max_tokens: int) -> str: ...


class RegexTokenCounter:
    """Deterministic offline budget counter; replaceable by a model tokenizer."""

    def count(self, text: str) -> int:
        return sum(1 for _ in _TOKEN.finditer(text))

    def truncate(self, text: str, max_tokens: int) -> str:
        if max_tokens < 1:
            return ""
        matches = list(_TOKEN.finditer(text))
        if len(matches) <= max_tokens:
            return text
        return text[: matches[max_tokens - 1].end()].rstrip()


class ContextItem(ContractModel):
    """Generation context tied to the authorized retrieval child that found it."""

    child: RetrievedChunk
    content: str = Field(min_length=1)
    used_parent: bool
    truncated: bool
    token_count: int = Field(ge=1)


class ExpandedContext(ContractModel):
    items: list[ContextItem] = Field(default_factory=list)
    total_tokens: int = Field(ge=0)
    token_budget: int = Field(ge=1)


class ParentContextExpander:
    """Expand ranked children to parents without crossing scope or budget."""

    def __init__(
        self,
        parent_store: ParentStore,
        *,
        token_budget: int = 1_200,
        max_parents: int = 8,
        token_counter: TokenCounter | None = None,
    ) -> None:
        if token_budget < 1 or max_parents < 1:
            raise ValueError("token_budget and max_parents must be positive")
        self._parent_store = parent_store
        self._token_budget = token_budget
        self._max_parents = max_parents
        self._token_counter = token_counter or RegexTokenCounter()

    async def expand(
        self,
        children: Sequence[RetrievedChunk],
        auth_filter: AuthFilter,
    ) -> ExpandedContext:
        selected = tuple(children[: self._max_parents])
        _validate_authorized_children(selected, auth_filter)
        if not selected:
            return ExpandedContext(total_tokens=0, token_budget=self._token_budget)

        requested_ids = tuple(dict.fromkeys(child.parent_id for child in selected))
        parents = await self._parent_store.get_parents(requested_ids, auth_filter)
        parent_by_id = _validate_and_index_parents(parents, selected, auth_filter)

        items: list[ContextItem] = []
        seen_parents: set[str] = set()
        seen_content: set[str] = set()
        remaining = self._token_budget
        for child in selected:
            if child.parent_id in seen_parents:
                continue
            seen_parents.add(child.parent_id)
            parent = parent_by_id.get(child.parent_id)
            content = child.content if parent is None else parent.content
            content_key = _SPACE.sub(" ", content).strip().casefold()
            if not content_key or content_key in seen_content:
                continue
            seen_content.add(content_key)
            original_tokens = self._token_counter.count(content)
            if original_tokens < 1 or remaining < 1:
                break
            bounded = self._token_counter.truncate(content, remaining)
            used_tokens = self._token_counter.count(bounded)
            if not bounded or used_tokens < 1:
                break
            items.append(
                ContextItem(
                    child=child.model_copy(deep=True),
                    content=bounded,
                    used_parent=parent is not None,
                    truncated=used_tokens < original_tokens,
                    token_count=used_tokens,
                )
            )
            remaining -= used_tokens
        return ExpandedContext(
            items=items,
            total_tokens=self._token_budget - remaining,
            token_budget=self._token_budget,
        )


class ContextPacker:
    """Select diverse, high-scoring parent evidence within a strict token budget."""

    def __init__(
        self,
        *,
        token_budget: int = 1_200,
        token_counter: TokenCounter | None = None,
    ) -> None:
        if token_budget < 1:
            raise ValueError("token_budget must be positive")
        self._token_budget = token_budget
        self._token_counter = token_counter or RegexTokenCounter()

    def pack(self, candidates: Sequence[ContextItem]) -> ExpandedContext:
        items: list[ContextItem] = []
        seen_chunks: set[tuple[str, int, str | None, str]] = set()
        seen_content: set[str] = set()
        remaining = self._token_budget
        for candidate in _diversified(candidates):
            child = candidate.child
            chunk_key = (
                child.document_id,
                child.document_version,
                child.index_version,
                child.chunk_id,
            )
            content_key = _SPACE.sub(" ", candidate.content).strip().casefold()
            if chunk_key in seen_chunks or not content_key or content_key in seen_content:
                continue
            original_tokens = self._token_counter.count(candidate.content)
            if original_tokens < 1 or remaining < 1:
                break
            bounded = self._token_counter.truncate(candidate.content, remaining)
            used_tokens = self._token_counter.count(bounded)
            if not bounded or used_tokens < 1:
                break
            seen_chunks.add(chunk_key)
            seen_content.add(content_key)
            items.append(
                ContextItem(
                    child=child.model_copy(deep=True),
                    content=bounded,
                    used_parent=candidate.used_parent,
                    truncated=candidate.truncated or used_tokens < original_tokens,
                    token_count=used_tokens,
                )
            )
            remaining -= used_tokens
        return ExpandedContext(
            items=items,
            total_tokens=self._token_budget - remaining,
            token_budget=self._token_budget,
        )


def _diversified(candidates: Sequence[ContextItem]) -> list[ContextItem]:
    ranked = sorted(
        enumerate(candidates),
        key=lambda pair: (-(pair[1].child.score or 0.0), pair[0]),
    )
    first_by_source: list[ContextItem] = []
    repeated_sources: list[ContextItem] = []
    seen_sources: set[str] = set()
    for _, candidate in ranked:
        source = candidate.child.source_uri
        if source in seen_sources:
            repeated_sources.append(candidate)
        else:
            seen_sources.add(source)
            first_by_source.append(candidate)
    return [*first_by_source, *repeated_sources]


def _validate_authorized_children(
    children: Sequence[RetrievedChunk], auth_filter: AuthFilter
) -> None:
    if not (
        auth_filter.allowed_document_ids
        and auth_filter.allowed_index_versions
        and auth_filter.allowed_document_versions
    ):
        if children:
            raise ParentAuthorizationError("authorization metadata is incomplete")
        return
    for child in children:
        if not _authorized(
            tenant_id=child.tenant_id,
            workspace_id=child.workspace_id,
            document_id=child.document_id,
            document_version=child.document_version,
            index_version=child.index_version,
            visibility=child.visibility,
            acl_groups=child.acl_groups,
            auth_filter=auth_filter,
        ):
            raise ParentAuthorizationError("child is outside authorization scope")


def _validate_and_index_parents(
    parents: Sequence[ParentChunk],
    children: Sequence[RetrievedChunk],
    auth_filter: AuthFilter,
) -> dict[str, ParentChunk]:
    children_by_parent: dict[str, list[RetrievedChunk]] = {}
    for child in children:
        children_by_parent.setdefault(child.parent_id, []).append(child)
    indexed: dict[str, ParentChunk] = {}
    for parent in parents:
        matching_children = children_by_parent.get(parent.parent_id)
        if not matching_children or parent.parent_id in indexed:
            raise ParentAuthorizationError("parent response is unrequested or duplicated")
        if not _authorized(
            tenant_id=parent.tenant_id,
            workspace_id=parent.workspace_id,
            document_id=parent.document_id,
            document_version=parent.document_version,
            index_version=parent.index_version,
            visibility=parent.visibility,
            acl_groups=parent.acl_groups,
            auth_filter=auth_filter,
        ):
            raise ParentAuthorizationError("parent is outside authorization scope")
        if not any(
            parent.document_id == child.document_id
            and parent.document_version == child.document_version
            and parent.index_version == child.index_version
            and parent.source_uri == child.source_uri
            for child in matching_children
        ):
            raise ParentAuthorizationError("parent lineage does not match its child")
        indexed[parent.parent_id] = parent
    return indexed


def _authorized(
    *,
    tenant_id: str,
    workspace_id: str,
    document_id: str,
    document_version: int,
    index_version: str | None,
    visibility: str,
    acl_groups: Sequence[str],
    auth_filter: AuthFilter,
) -> bool:
    binding = (document_id, document_version, index_version)
    visible = visibility == "tenant" or bool(set(acl_groups).intersection(auth_filter.group_ids))
    return bool(
        tenant_id == auth_filter.tenant_id
        and workspace_id == auth_filter.workspace_id
        and document_id in auth_filter.allowed_document_ids
        and index_version in auth_filter.allowed_index_versions
        and binding in auth_filter.allowed_document_versions
        and visible
    )
