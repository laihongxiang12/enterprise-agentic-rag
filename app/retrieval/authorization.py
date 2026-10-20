"""Shared authorization checks for chunks returned by retrieval adapters."""

from app.schemas.retrieval import AuthFilter, RetrievedChunk


def chunk_matches_scope(chunk: RetrievedChunk, auth_filter: AuthFilter) -> bool:
    """Revalidate a retrieved chunk against the complete trusted scope."""
    binding = (chunk.document_id, chunk.document_version, chunk.index_version)
    visible = chunk.visibility == "tenant" or bool(
        set(chunk.acl_groups).intersection(auth_filter.group_ids)
    )
    return bool(
        auth_filter.allowed_document_ids
        and auth_filter.allowed_index_versions
        and auth_filter.allowed_document_versions
        and chunk.tenant_id == auth_filter.tenant_id
        and chunk.workspace_id == auth_filter.workspace_id
        and chunk.document_id in auth_filter.allowed_document_ids
        and chunk.index_version in auth_filter.allowed_index_versions
        and binding in auth_filter.allowed_document_versions
        and visible
    )
