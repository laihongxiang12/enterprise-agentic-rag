"""Identity and document metadata contracts."""

from typing import Annotated, Literal

from pydantic import Field

from app.schemas.base import ContractModel

NonEmptyString = Annotated[str, Field(min_length=1)]


class Principal(ContractModel):
    """Verified caller identity created only at the authentication boundary."""

    user_id: NonEmptyString
    tenant_id: NonEmptyString
    group_ids: list[str] = Field(default_factory=list)
    scopes: list[str] = Field(default_factory=list)


class DocumentMeta(ContractModel):
    """Versioned document metadata stored outside retrieval content."""

    tenant_id: NonEmptyString
    workspace_id: NonEmptyString
    document_id: NonEmptyString
    version: int = Field(ge=1)
    title: NonEmptyString
    source_uri: NonEmptyString
    source_type: Literal["pdf", "markdown", "html"]
    visibility: Literal["restricted", "tenant"] = "restricted"
    acl_groups: list[str] = Field(default_factory=list)
    content_hash: NonEmptyString
    acl_version: int = Field(ge=1)
    index_version: NonEmptyString
    status: Literal["pending", "processing", "ready", "failed", "deleted"]
