"""Framework-neutral query orchestration boundary."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Literal, Protocol, cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from pydantic import TypeAdapter, ValidationError

from app.agents.graph import build_agent_graph
from app.agents.state import AgentRuntimeContext, build_initial_state
from app.core.errors import AppError
from app.retrieval.base import Retriever
from app.schemas.agent import AgentState
from app.schemas.api import Citation, QueryRequest, QueryResponse, Usage
from app.schemas.documents import Principal
from app.schemas.retrieval import AuthFilter, RetrievedChunk
from app.stores.base import CacheStore
from app.stores.checkpoint import checkpoint_config
from app.stores.redis import RetrievalCacheKey

_CHUNK_LIST = TypeAdapter(list[RetrievedChunk])


class RuntimeContextFactory(Protocol):
    async def __call__(
        self, principal: Principal, workspace_id: str | None
    ) -> AgentRuntimeContext: ...


class QueryService:
    def __init__(
        self,
        *,
        checkpointer: BaseCheckpointSaver[str] | None = None,
        max_concurrency: int = 8,
        concurrency_wait_seconds: float = 0.1,
    ) -> None:
        if max_concurrency < 1 or concurrency_wait_seconds <= 0:
            raise ValueError("query concurrency settings must be positive")
        self._graph = build_agent_graph(checkpointer)
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._concurrency_wait_seconds = concurrency_wait_seconds

    async def execute(
        self,
        request: QueryRequest,
        context: AgentRuntimeContext,
        *,
        trace_id: str,
    ) -> AgentState:
        """Run one query with identity and authorization from trusted context only."""
        state = build_initial_state(
            context,
            trace_id=trace_id,
            query=request.query,
            max_retries=request.max_retries,
        )
        if request.session_id is not None:
            state["session_id"] = request.session_id
        config = checkpoint_config(
            context.principal,
            workspace_id=context.auth_filter.workspace_id,
            session_id=request.session_id or trace_id,
        )
        try:
            async with asyncio.timeout(self._concurrency_wait_seconds):
                await self._semaphore.acquire()
        except TimeoutError as exc:
            raise AppError(
                code="query_concurrency_exceeded",
                message="Too many queries are running concurrently",
                status_code=429,
            ) from exc
        try:
            result = await self._graph.ainvoke(state, config=config, context=context)
            return cast(AgentState, result)
        finally:
            self._semaphore.release()


class CachedRetriever:
    """Cache only authorized retrieval chunks; generated answers are never cached."""

    def __init__(
        self,
        retriever: Retriever,
        cache: CacheStore,
        *,
        ttl_seconds: int,
        config_hash: str,
    ) -> None:
        if ttl_seconds < 1 or not config_hash.strip():
            raise ValueError("cache TTL and config_hash must be configured")
        self._retriever = retriever
        self._cache = cache
        self._ttl_seconds = ttl_seconds
        self._config_hash = config_hash

    async def aretrieve(
        self,
        query: str,
        auth_filter: AuthFilter,
        top_k: int,
    ) -> list[RetrievedChunk]:
        key = RetrievalCacheKey.from_request(
            auth_filter,
            query,
            config_hash=self._config_hash,
            top_k=top_k,
        )
        cached = await self._cache.get(key.redis_key)
        if cached is not None:
            try:
                return _CHUNK_LIST.validate_json(cached)
            except ValidationError:
                await self._cache.invalidate(key.redis_key)
        chunks = await self._retriever.aretrieve(query, auth_filter, top_k)
        encoded = json.dumps(
            [chunk.model_dump(mode="json") for chunk in chunks],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
        await self._cache.set(key.redis_key, encoded, self._ttl_seconds)
        return chunks


class AgentQueryApplicationService:
    """Convert graph state into the stable public query response contract."""

    def __init__(
        self,
        context_factory: RuntimeContextFactory,
        graph_service: QueryService | None = None,
    ) -> None:
        self._context_factory = context_factory
        self._graph_service = graph_service or QueryService()

    async def query(
        self,
        request: QueryRequest,
        principal: Principal,
        *,
        trace_id: str,
    ) -> QueryResponse:
        if "query:read" not in principal.scopes:
            raise AppError(code="forbidden", message="Required scope is missing", status_code=403)
        started = time.perf_counter()
        context = await self._context_factory(principal, request.workspace_id)
        state = await self._graph_service.execute(request, context, trace_id=trace_id)
        response = state.get("response", {})
        action = response.get("action")
        raw_reason = response.get("finish_reason")
        allowed = {"completed", "insufficient_evidence", "policy_blocked", "error"}
        if raw_reason in allowed:
            finish_reason: Literal[
                "completed", "insufficient_evidence", "policy_blocked", "error"
            ] = cast(
                Literal["completed", "insufficient_evidence", "policy_blocked", "error"],
                raw_reason,
            )
        elif action == "policy_blocked":
            finish_reason = "policy_blocked"
        elif action == "direct":
            finish_reason = "completed"
        else:
            finish_reason = "error"

        answer = response.get("answer")
        if not isinstance(answer, str):
            answer = "" if finish_reason != "policy_blocked" else "Request blocked by policy."
        raw_citations = response.get("citations", [])
        citations = (
            [Citation.model_validate(item) for item in raw_citations]
            if isinstance(raw_citations, list)
            else []
        )
        latency_ms = max(0, int((time.perf_counter() - started) * 1_000))
        return QueryResponse(
            trace_id=trace_id,
            answer=answer,
            citations=citations,
            usage=Usage(),
            latency_ms=latency_ms,
            retries=state.get("retry_count", 0),
            finish_reason=finish_reason,
        )
