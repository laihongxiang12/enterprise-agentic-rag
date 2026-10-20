"""Framework-neutral, fail-closed service behind the MCP tools."""

from __future__ import annotations

import asyncio

from pydantic import Field

from app.api.dependencies import RequestLimiter
from app.core.dependencies import PolicyEngine
from app.core.errors import AppError
from app.retrieval.authorization import chunk_matches_scope
from app.retrieval.base import Retriever
from app.schemas.base import ContractModel
from app.schemas.documents import Principal
from app.stores.base import MetadataStore


class SearchHit(ContractModel):
    document_id: str
    document_version: int = Field(ge=1)
    chunk_id: str
    title: str
    source_uri: str
    snippet: str
    page: int | None = None
    score: float | None = None


class SearchDocsResult(ContractModel):
    items: list[SearchHit] = Field(default_factory=list)
    count: int = Field(ge=0)


class GetDocResult(ContractModel):
    document_id: str
    document_version: int = Field(ge=1)
    workspace_id: str
    title: str
    source_uri: str
    source_type: str
    visibility: str
    status: str


class MCPServiceError(RuntimeError):
    """Stable tool error that never contains dependency exception details."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class MCPReadService:
    """Expose only authorized retrieval and metadata reads."""

    def __init__(
        self,
        *,
        metadata: MetadataStore,
        retriever: Retriever,
        policy: PolicyEngine,
        limiter: RequestLimiter,
        max_limit: int = 20,
        timeout_seconds: float = 5.0,
        max_snippet_chars: int = 500,
    ) -> None:
        if max_limit < 1 or timeout_seconds <= 0 or max_snippet_chars < 1:
            raise ValueError("MCP service limits must be positive")
        self._metadata = metadata
        self._retriever = retriever
        self._policy = policy
        self._limiter = limiter
        self._max_limit = max_limit
        self._timeout_seconds = timeout_seconds
        self._max_snippet_chars = max_snippet_chars

    async def search_docs(
        self,
        principal: Principal,
        workspace_id: str,
        query: str,
        limit: int,
    ) -> SearchDocsResult:
        if not self._policy.authorize_tool(principal, "search_docs"):
            raise MCPServiceError("forbidden")
        normalized_query = query.strip()
        if not normalized_query or not 1 <= limit <= self._max_limit:
            raise MCPServiceError("invalid_request")
        try:
            async with asyncio.timeout(self._timeout_seconds):
                await self._limiter.check(principal, "mcp:search_docs")
                scope = await self._metadata.authorization_scope(principal, workspace_id)
                auth_filter = scope.to_auth_filter()
                chunks = await self._retriever.aretrieve(
                    normalized_query,
                    auth_filter,
                    limit,
                )
        except TimeoutError as exc:
            raise MCPServiceError("tool_timeout") from exc
        except AppError as exc:
            raise MCPServiceError(exc.code) from exc
        except MCPServiceError:
            raise
        except Exception as exc:
            raise MCPServiceError("search_unavailable") from exc
        if any(not chunk_matches_scope(chunk, auth_filter) for chunk in chunks):
            raise MCPServiceError("retrieval_scope_violation")
        items = [
            SearchHit(
                document_id=chunk.document_id,
                document_version=chunk.document_version,
                chunk_id=chunk.chunk_id,
                title=chunk.title,
                source_uri=chunk.source_uri,
                snippet=chunk.content[: self._max_snippet_chars],
                page=chunk.page,
                score=chunk.score,
            )
            for chunk in chunks
        ]
        return SearchDocsResult(items=items, count=len(items))

    async def get_doc(
        self,
        principal: Principal,
        workspace_id: str,
        document_id: str,
    ) -> GetDocResult:
        if not self._policy.authorize_tool(principal, "get_doc"):
            raise MCPServiceError("forbidden")
        normalized_id = document_id.strip()
        if not normalized_id or len(normalized_id) > 256:
            raise MCPServiceError("invalid_request")
        try:
            async with asyncio.timeout(self._timeout_seconds):
                await self._limiter.check(principal, "mcp:get_doc")
                document = await self._metadata.get_authorized_document(
                    principal,
                    workspace_id,
                    normalized_id,
                )
        except TimeoutError as exc:
            raise MCPServiceError("tool_timeout") from exc
        except AppError as exc:
            raise MCPServiceError(exc.code) from exc
        except Exception as exc:
            raise MCPServiceError("document_unavailable") from exc
        if document is None:
            raise MCPServiceError("not_found")
        return GetDocResult(
            document_id=document.document_id,
            document_version=document.version,
            workspace_id=document.workspace_id,
            title=document.title,
            source_uri=document.source_uri,
            source_type=document.source_type,
            visibility=document.visibility,
            status=document.status,
        )
