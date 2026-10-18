"""Framework-neutral query orchestration boundary."""

from __future__ import annotations

import time
from typing import Literal, Protocol, cast

from app.agents.graph import build_agent_graph
from app.agents.state import AgentRuntimeContext, build_initial_state
from app.core.errors import AppError
from app.schemas.agent import AgentState
from app.schemas.api import Citation, QueryRequest, QueryResponse, Usage
from app.schemas.documents import Principal


class RuntimeContextFactory(Protocol):
    async def __call__(
        self, principal: Principal, workspace_id: str | None
    ) -> AgentRuntimeContext: ...


class QueryService:
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
        result = await build_agent_graph().ainvoke(state, context=context)
        return cast(AgentState, result)


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
