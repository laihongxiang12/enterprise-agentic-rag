"""Explicitly local Fake runtime for a real MCP stdio handshake."""

from __future__ import annotations

import os

from pydantic import SecretStr

from app.api.dependencies import NoopRequestLimiter
from app.core.config import Settings
from app.core.security import BasicPolicyEngine, DevTokenService
from app.mcp.server import MCPConnectionContext
from app.mcp.service import MCPReadService
from app.retrieval.fakes import FakeEmbedding, FakeVectorStore
from app.retrieval.hybrid import HybridRetriever
from app.schemas.documents import DocumentMeta, Principal
from app.schemas.retrieval import RetrievedChunk
from app.stores.fakes import MemoryAuthorizationMetadataStore

DEMO_WORKSPACE = "policies"
DEMO_DOCUMENT_ID = "demo-annual-leave"


async def build_demo_connection(
    principal: Principal,
    *,
    workspace_id: str = DEMO_WORKSPACE,
    timeout_seconds: float = 5.0,
) -> MCPConnectionContext:
    """Build a deterministic fixture that still exercises ACL pre-filter and RRF."""
    metadata = MemoryAuthorizationMetadataStore()
    await metadata.put_workspace_membership(principal.tenant_id, workspace_id, principal.user_id)
    document = DocumentMeta(
        tenant_id=principal.tenant_id,
        workspace_id=workspace_id,
        document_id=DEMO_DOCUMENT_ID,
        version=1,
        title="Northstar 年假申请制度",
        source_uri="demo://northstar/hr/annual-leave",
        source_type="markdown",
        visibility="restricted",
        acl_groups=["hr"],
        content_hash="a" * 64,
        acl_version=1,
        index_version="mcp-demo-v1",
        status="ready",
    )
    await metadata.put_document(document)
    chunk = RetrievedChunk(
        tenant_id=principal.tenant_id,
        workspace_id=workspace_id,
        document_id=document.document_id,
        document_version=document.version,
        chunk_id="demo-annual-leave-c1",
        parent_id="demo-annual-leave-p1",
        content="年假应至少提前三个工作日提交, 直属经理应在两个工作日内审批。",
        title=document.title,
        source_uri=document.source_uri,
        visibility=document.visibility,
        acl_groups=list(document.acl_groups),
        index_version=document.index_version,
    )
    embedding = FakeEmbedding(dimension=64)
    store = FakeVectorStore()
    await store.upsert(
        [chunk],
        await embedding.embed_documents([chunk.content]),
        document.index_version,
    )
    service = MCPReadService(
        metadata=metadata,
        retriever=HybridRetriever(store, embedding),
        policy=BasicPolicyEngine(),
        limiter=NoopRequestLimiter(),
        timeout_seconds=timeout_seconds,
    )
    return MCPConnectionContext(
        principal=principal,
        workspace_id=workspace_id,
        service=service,
    )


async def build_demo_connection_from_environment() -> MCPConnectionContext:
    """Verify a signed local token before constructing the stdio context."""
    secret = os.getenv("DEV_AUTH_SECRET")
    token = os.getenv("MCP_DEV_BEARER_TOKEN")
    workspace_id = os.getenv("MCP_WORKSPACE_ID", DEMO_WORKSPACE).strip()
    if not secret or not token or not workspace_id or len(workspace_id) > 128:
        raise RuntimeError("MCP local authentication configuration is missing or invalid")
    settings = Settings(
        app_env="local",
        app_mode="fake",
        bind_host="127.0.0.1",
        dev_auth_enabled=True,
        dev_auth_secret=SecretStr(secret),
        dev_auth_issuer=os.getenv("DEV_AUTH_ISSUER", "enterprise-agentic-rag-dev"),
        dev_auth_audience=os.getenv("DEV_AUTH_AUDIENCE", "enterprise-agentic-rag"),
        _env_file=None,
    )
    principal = DevTokenService(settings).verify(token)
    return await build_demo_connection(principal, workspace_id=workspace_id)
