"""Authorized retrieval contracts."""

from typing import Annotated

from pydantic import ConfigDict, Field

from app.schemas.base import ContractModel

NonEmptyString = Annotated[str, Field(min_length=1)]


class AuthFilter(ContractModel):
    """Trusted, immutable filter passed to every vector-store search.

    An empty ``allowed_document_ids`` set is intentionally fail-closed.
    This object is built from a verified Principal and metadata policy, never
    from an LLM response or a public request body.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        frozen=True,
    )

    tenant_id: NonEmptyString
    workspace_id: NonEmptyString
    group_ids: frozenset[str] = Field(default_factory=frozenset)
    allowed_document_ids: frozenset[str] = Field(default_factory=frozenset)


class RetrievedChunk(ContractModel):
    """A chunk that has already passed storage-side authorization."""

    tenant_id: NonEmptyString
    workspace_id: NonEmptyString
    document_id: NonEmptyString
    document_version: int = Field(ge=1)
    chunk_id: NonEmptyString
    parent_id: NonEmptyString
    content: NonEmptyString
    title: NonEmptyString
    source_uri: NonEmptyString
    page: int | None = None
    heading_path: list[str] = Field(default_factory=list)
    score: float | None = None
