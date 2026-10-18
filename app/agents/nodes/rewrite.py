"""One bounded structured query rewrite before each additional retrieval."""

import json
from typing import Literal

from langgraph.runtime import Runtime

from app.agents.state import AgentRuntimeContext, append_error
from app.llm.base import LLMError, LLMMessage
from app.schemas.agent import AgentState, EvidenceGrade, QueryRewrite

AfterRewrite = Literal["retrieve", "insufficient_evidence"]


async def rewrite_query(
    state: AgentState,
    runtime: Runtime[AgentRuntimeContext],
) -> AgentState:
    context = runtime.context
    if context is None:
        return _failed(state, "missing_runtime_context")
    remaining_milliseconds = state.get("deadline_epoch_ms", 0) - context.now_epoch_ms()
    if remaining_milliseconds <= 0:
        return _failed(state, "deadline_exceeded")
    if (
        state.get("retry_count", 0) >= state.get("max_retries", 0)
        or state.get("remaining_tool_calls", 0) <= 0
        or state.get("remaining_agent_rounds", 0) <= 0
    ):
        return _failed(state, "retry_budget_exhausted")

    try:
        grade = EvidenceGrade.model_validate(state.get("grade", {}))
        result = await context.llm.complete(
            [
                LLMMessage(
                    role="system",
                    content=(
                        "Rewrite the query to address missing evidence. Preserve intent and "
                        "never add tenant, identity, permissions, SQL, web, or tool requests."
                    ),
                ),
                LLMMessage(
                    role="user",
                    content=rewrite_request_content(
                        state.get("active_query", ""),
                        grade,
                    ),
                ),
            ],
            QueryRewrite,
            timeout_seconds=min(
                context.limits.rewrite_timeout_seconds,
                remaining_milliseconds / 1_000,
            ),
        )
        if not isinstance(result.parsed, QueryRewrite):
            raise LLMError("rewriter returned no validated QueryRewrite")
        rewritten = result.parsed.query.strip()
    except (LLMError, ValueError):
        return _failed(state, "rewrite_failed")

    current = state.get("active_query", "")
    if _normalized(rewritten) == _normalized(current):
        return _failed(state, "rewrite_unchanged")
    return AgentState(
        active_query=rewritten,
        plan=[*state.get("plan", []), rewritten],
        retry_count=state.get("retry_count", 0) + 1,
        remaining_agent_rounds=state.get("remaining_agent_rounds", 0) - 1,
        response={"action": "rewritten"},
    )


def rewrite_request_content(query: str, grade: EvidenceGrade) -> str:
    return json.dumps(
        {
            "query": query,
            "missing_aspects": grade.missing_aspects,
            "grader_reason": grade.reason,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def select_after_rewrite(state: AgentState) -> AfterRewrite:
    return (
        "retrieve"
        if state.get("response", {}).get("action") == "rewritten"
        else "insufficient_evidence"
    )


def _normalized(value: str) -> str:
    return " ".join(value.casefold().split())


def _failed(state: AgentState, code: str) -> AgentState:
    return AgentState(
        response={"action": "rewrite_failed"},
        error_codes=append_error(state, code),
    )
