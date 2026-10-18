"""Terminal Stage 11 direct, retrieval, and policy-blocked nodes."""

from langgraph.runtime import Runtime

from app.agents.state import AgentRuntimeContext, append_error, blocked_update
from app.schemas.agent import AgentState


def direct_response(state: AgentState) -> AgentState:
    return AgentState(
        response={
            "action": "direct",
            "route_reason": state.get("route_reason", ""),
            "generation": "deferred_to_stage_14",
        }
    )


async def retrieve(
    state: AgentState,
    runtime: Runtime[AgentRuntimeContext],
) -> AgentState:
    context = runtime.context
    if context is None:
        return blocked_update(state, "missing_runtime_context", "trusted context is required")
    if state.get("deadline_epoch_ms", 0) <= context.now_epoch_ms():
        update = blocked_update(state, "deadline_exceeded", "agent deadline exceeded")
        update["response"] = {"action": "policy_blocked"}
        return update
    remaining_tool_calls = state.get("remaining_tool_calls", 0)
    if remaining_tool_calls <= 0:
        update = blocked_update(state, "tool_budget_exhausted", "tool budget exhausted")
        update["response"] = {"action": "policy_blocked"}
        return update

    chunks = await context.retriever.aretrieve(
        state.get("active_query", ""),
        context.auth_filter,
        context.limits.retrieval_top_k,
    )
    return AgentState(
        retrieved_chunks=[chunk.model_dump(mode="json") for chunk in chunks],
        remaining_tool_calls=remaining_tool_calls - 1,
        response={"action": "retrieved", "count": len(chunks)},
    )


def policy_blocked(state: AgentState) -> AgentState:
    errors = state.get("error_codes", [])
    if not errors:
        errors = append_error(state, "policy_blocked")
    return AgentState(
        response={
            "action": "policy_blocked",
            "reason": state.get("route_reason", "request blocked by policy"),
        },
        error_codes=errors,
    )
