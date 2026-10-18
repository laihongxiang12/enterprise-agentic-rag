"""Structured query analysis with deterministic safety escalation."""

from langgraph.runtime import Runtime

from app.agents.state import AgentRuntimeContext, append_error, blocked_update
from app.llm.base import LLMError, LLMMessage
from app.schemas.agent import AgentState, RouteDecision

_INTERNAL_FACT_TERMS = (
    "公司",
    "内部",
    "员工",
    "制度",
    "政策",
    "年假",
    "报销",
    "薪酬",
    "我们的",
    "our company",
    "our policy",
    "internal",
    "employee policy",
)


def is_internal_fact_query(query: str) -> bool:
    normalized = query.casefold()
    return any(term in normalized for term in _INTERNAL_FACT_TERMS)


async def analyze_and_route(
    state: AgentState,
    runtime: Runtime[AgentRuntimeContext],
) -> AgentState:
    context = runtime.context
    if context is None:
        return blocked_update(state, "missing_runtime_context", "trusted context is required")
    remaining_milliseconds = state.get("deadline_epoch_ms", 0) - context.now_epoch_ms()
    if remaining_milliseconds <= 0:
        return blocked_update(state, "deadline_exceeded", "agent deadline exceeded")
    remaining_rounds = state.get("remaining_agent_rounds", 0)
    if remaining_rounds <= 0:
        return blocked_update(state, "round_budget_exhausted", "agent round budget exhausted")
    query = state.get("active_query", "").strip()
    if not query:
        return blocked_update(state, "empty_query", "query must not be empty")

    errors = state.get("error_codes", [])
    try:
        result = await context.llm.complete(
            [
                LLMMessage(
                    role="system",
                    content=(
                        "Classify the user query as direct, knowledge, sql, or web. "
                        "Requests about company facts or policies must use knowledge."
                    ),
                ),
                LLMMessage(role="user", content=query),
            ],
            RouteDecision,
            timeout_seconds=min(
                context.limits.route_timeout_seconds,
                remaining_milliseconds / 1_000,
            ),
        )
        if not isinstance(result.parsed, RouteDecision):
            raise LLMError("router returned no validated RouteDecision")
        decision = result.parsed
        route = decision.route
        confidence = decision.confidence
        reason = decision.reason
    except LLMError:
        route = "knowledge"
        confidence = 0.0
        reason = "router failed validation; fail-safe knowledge retrieval"
        errors = append_error(state, "router_fallback")

    if route == "direct" and (
        is_internal_fact_query(query) or confidence < context.limits.route_min_confidence
    ):
        route = "knowledge"
        reason = "direct route overridden by deterministic knowledge-safety rule"

    return AgentState(
        route=route,
        route_confidence=confidence,
        route_reason=reason,
        remaining_agent_rounds=remaining_rounds - 1,
        error_codes=errors,
    )
