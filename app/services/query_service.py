"""Framework-neutral query orchestration boundary."""

from typing import cast

from app.agents.graph import build_agent_graph
from app.agents.state import AgentRuntimeContext, build_initial_state
from app.schemas.agent import AgentState
from app.schemas.api import QueryRequest


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
