"""MCP service timeout, tenant isolation, and result revalidation tests."""

import asyncio

import pytest

from app.api.dependencies import NoopRequestLimiter
from app.core.security import BasicPolicyEngine
from app.mcp.service import MCPReadService, MCPServiceError
from app.retrieval.base import Retriever
from app.retrieval.fakes import FakeRetriever
from app.schemas.documents import DocumentMeta, Principal
from app.schemas.retrieval import AuthFilter, RetrievedChunk
from app.stores.fakes import MemoryAuthorizationMetadataStore


def user(*, tenant: str = "tenant-a") -> Principal:
    return Principal(
        user_id="alice",
        tenant_id=tenant,
        group_ids=["hr"],
        scopes=["tools:get_doc", "tools:search_docs"],
    )


def document(tenant: str, document_id: str) -> DocumentMeta:
    return DocumentMeta(
        tenant_id=tenant,
        workspace_id="hr",
        document_id=document_id,
        version=1,
        title=f"Policy {document_id}",
        source_uri=f"fixture:///{tenant}/{document_id}.md",
        source_type="markdown",
        visibility="restricted",
        acl_groups=["hr"],
        content_hash="a" * 64,
        acl_version=1,
        index_version=f"index-{document_id}",
        status="ready",
    )


def chunk(tenant: str, document_id: str) -> RetrievedChunk:
    return RetrievedChunk(
        tenant_id=tenant,
        workspace_id="hr",
        document_id=document_id,
        document_version=1,
        chunk_id=f"chunk-{document_id}",
        parent_id=f"parent-{document_id}",
        content=f"Secret for {tenant}",
        title=f"Policy {document_id}",
        source_uri=f"fixture:///{tenant}/{document_id}.md",
        visibility="restricted",
        acl_groups=["hr"],
        index_version=f"index-{document_id}",
    )


async def store_for_both_tenants() -> MemoryAuthorizationMetadataStore:
    store = MemoryAuthorizationMetadataStore()
    await store.put_workspace_membership("tenant-a", "hr", "alice")
    await store.put_workspace_membership("tenant-b", "hr", "bob")
    await store.put_document(document("tenant-a", "allowed"))
    await store.put_document(document("tenant-b", "secret"))
    return store


async def test_cross_tenant_get_doc_is_indistinguishable_from_missing() -> None:
    service = MCPReadService(
        metadata=await store_for_both_tenants(),
        retriever=FakeRetriever(),
        policy=BasicPolicyEngine(),
        limiter=NoopRequestLimiter(),
    )

    with pytest.raises(MCPServiceError, match="not_found"):
        await service.get_doc(user(), "hr", "secret")


async def test_out_of_scope_retriever_result_fails_closed() -> None:
    leaked = chunk("tenant-b", "secret")
    service = MCPReadService(
        metadata=await store_for_both_tenants(),
        retriever=FakeRetriever({"policy": [leaked]}),
        policy=BasicPolicyEngine(),
        limiter=NoopRequestLimiter(),
    )

    with pytest.raises(MCPServiceError, match="retrieval_scope_violation"):
        await service.search_docs(user(), "hr", "policy", 3)


class SlowRetriever(Retriever):
    async def aretrieve(
        self,
        query: str,
        auth_filter: AuthFilter,
        top_k: int,
    ) -> list[RetrievedChunk]:
        del query, auth_filter, top_k
        await asyncio.sleep(1)
        return []


async def test_tool_timeout_is_bounded_and_sanitized() -> None:
    service = MCPReadService(
        metadata=await store_for_both_tenants(),
        retriever=SlowRetriever(),
        policy=BasicPolicyEngine(),
        limiter=NoopRequestLimiter(),
        timeout_seconds=0.01,
    )

    with pytest.raises(MCPServiceError, match="tool_timeout") as captured:
        await service.search_docs(user(), "hr", "policy", 3)

    assert "sleep" not in str(captured.value)
