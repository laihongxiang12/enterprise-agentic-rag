"""Server-side policy checks that cannot be overridden by model output."""

from typing import Literal

from langgraph.runtime import Runtime

from app.agents.nodes.analyze import is_internal_fact_query
from app.agents.state import AgentRuntimeContext, blocked_update
from app.schemas.agent import AgentState

NextNode = Literal["direct", "retrieve", "policy_blocked"]


def policy_check(
    state: AgentState,
    runtime: Runtime[AgentRuntimeContext],
) -> AgentState:
    context = runtime.context
    if context is None:
        return blocked_update(state, "missing_runtime_context", "trusted context is required")
    if state.get("deadline_epoch_ms", 0) <= context.now_epoch_ms():
        return blocked_update(state, "deadline_exceeded", "agent deadline exceeded")
    if (
        state.get("tenant_id") != context.principal.tenant_id
        or state.get("user_id") != context.principal.user_id
    ):
        return blocked_update(state, "identity_scope_mismatch", "identity scope mismatch")

    route = state.get("route")
    query = state.get("active_query", "")
    if route == "direct":
        if is_internal_fact_query(query):
            return blocked_update(
                state,
                "direct_internal_fact_blocked",
                "internal facts cannot bypass retrieval",
            )
        return AgentState()

    if route == "knowledge":
        if state.get("remaining_tool_calls", 0) <= 0:
            return blocked_update(state, "tool_budget_exhausted", "tool budget exhausted")
        if (
            state.get("tenant_id") != context.auth_filter.tenant_id
            or state.get("workspace_id") != context.auth_filter.workspace_id
        ):
            return blocked_update(
                state,
                "retrieval_scope_mismatch",
                "retrieval authorization scope mismatch",
            )
        if not (
            context.auth_filter.allowed_document_ids
            and context.auth_filter.allowed_index_versions
            and context.auth_filter.allowed_document_versions
        ):
            return blocked_update(
                state,
                "empty_authorization_scope",
                "retrieval authorization scope is incomplete",
            )
        return AgentState()

    if route in {"sql", "web"}:
        authorized = context.policy_engine.authorize_tool(context.principal, route)
        enabled = context.limits.sql_enabled if route == "sql" else context.limits.web_enabled
        if not enabled:
            return blocked_update(state, f"{route}_disabled", f"{route} tool is disabled")
        if not authorized:
            return blocked_update(
                state,
                f"{route}_unauthorized",
                f"{route} tool is not authorized",
            )
        return blocked_update(
            state,
            f"{route}_not_implemented",
            f"{route} tool is not implemented in stage 11",
        )

    return blocked_update(state, "unknown_route", "router produced an unknown route")


def select_next_node(state: AgentState) -> NextNode:
    route = state.get("route")
    if route == "direct":
        return "direct"
    if route == "knowledge":
        return "retrieve"
    return "policy_blocked"
