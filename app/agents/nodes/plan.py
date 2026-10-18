"""Structured multi-hop planning and bounded concurrent authorized retrieval."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from typing import Literal

from langgraph.runtime import Runtime
from pydantic import ValidationError

from app.agents.nodes.retrieve import is_authorized_chunk
from app.agents.state import AgentRuntimeContext, append_error
from app.llm.base import LLMError, LLMMessage
from app.retrieval.context import ContextItem, ContextPacker
from app.schemas.agent import AgentState, EvidenceGrade, MultiHopPlan
from app.schemas.retrieval import RetrievedChunk

AfterPlan = Literal["grade_evidence", "insufficient_evidence"]


async def plan_and_retrieve(
    state: AgentState,
    runtime: Runtime[AgentRuntimeContext],
) -> AgentState:
    context = runtime.context
    if context is None:
        return _failed(state, "missing_runtime_context", "trusted context is required")
    remaining_milliseconds = state.get("deadline_epoch_ms", 0) - context.now_epoch_ms()
    if remaining_milliseconds <= 0:
        return _failed(state, "deadline_exceeded", "agent deadline exceeded")
    if (
        state.get("retry_count", 0) >= state.get("max_retries", 0)
        or state.get("remaining_tool_calls", 0) < 2
        or state.get("remaining_agent_rounds", 0) <= 0
    ):
        return _failed(state, "retry_budget_exhausted", "multi-hop budget exhausted")

    try:
        grade = EvidenceGrade.model_validate(state.get("grade", {}))
        if grade.recommended_action != "decompose":
            raise ValueError("grader did not request decomposition")
        result = await context.llm.complete(
            [
                LLMMessage(
                    role="system",
                    content=(
                        "Decompose the query into two to four independent knowledge-base "
                        "subqueries. Preserve intent. Never add or change tenant, workspace, "
                        "identity, permissions, SQL, web, or tool scope."
                    ),
                ),
                LLMMessage(
                    role="user",
                    content=_plan_request_content(
                        state.get("active_query", ""),
                        grade,
                    ),
                ),
            ],
            MultiHopPlan,
            timeout_seconds=min(
                context.limits.planner_timeout_seconds,
                remaining_milliseconds / 1_000,
            ),
        )
        if not isinstance(result.parsed, MultiHopPlan):
            raise LLMError("planner returned no validated MultiHopPlan")
        plan = result.parsed
    except (LLMError, ValidationError, ValueError):
        return _failed(state, "planner_failed", "multi-hop planner failed safely")

    if len(plan.subqueries) > context.limits.max_subqueries:
        return _failed(state, "subquery_limit_exceeded", "subquery limit exceeded")
    tool_calls = len(plan.subqueries)
    if tool_calls > state.get("remaining_tool_calls", 0):
        return _failed(state, "tool_budget_exhausted", "tool budget exhausted")

    remaining_seconds = (state.get("deadline_epoch_ms", 0) - context.now_epoch_ms()) / 1_000
    if remaining_seconds <= 0:
        return _failed(state, "deadline_exceeded", "agent deadline exceeded")
    semaphore = asyncio.Semaphore(context.limits.subquery_concurrency)
    tasks = [
        asyncio.create_task(_retrieve_one(query, context, semaphore)) for query in plan.subqueries
    ]
    try:
        responses = await asyncio.wait_for(
            asyncio.gather(*tasks),
            timeout=remaining_seconds,
        )
    except TimeoutError:
        await _cancel(tasks)
        return _failed(
            state,
            "multihop_timeout",
            "multi-hop retrieval exceeded the deadline",
            tool_calls=tool_calls,
            clear_evidence=True,
        )
    except Exception:
        await _cancel(tasks)
        return _failed(
            state,
            "multihop_retrieval_failed",
            "multi-hop retrieval failed safely",
            tool_calls=tool_calls,
            clear_evidence=True,
        )

    if state.get("deadline_epoch_ms", 0) <= context.now_epoch_ms():
        return _failed(
            state,
            "deadline_exceeded",
            "agent deadline exceeded",
            tool_calls=tool_calls,
            clear_evidence=True,
        )
    existing = _state_chunks(state)
    if (
        existing is None
        or any(not is_authorized_chunk(chunk, context) for chunk in existing)
        or any(not is_authorized_chunk(chunk, context) for chunks in responses for chunk in chunks)
    ):
        return _failed(
            state,
            "unauthorized_evidence",
            "subquery evidence authorization failed",
            tool_calls=tool_calls,
            clear_evidence=True,
        )

    merged = _deduplicate([*existing, *(chunk for chunks in responses for chunk in chunks)])
    packed = ContextPacker(token_budget=context.limits.context_token_budget).pack(
        [
            ContextItem(
                child=chunk,
                content=chunk.content,
                used_parent=False,
                truncated=False,
                token_count=max(1, len(chunk.content)),
            )
            for chunk in merged
        ]
    )
    packed_chunks = [
        item.child.model_copy(update={"content": item.content}, deep=True) for item in packed.items
    ]
    packed_keys = {_chunk_key(chunk) for chunk in packed_chunks}
    coverage = {
        query: [chunk.chunk_id for chunk in chunks if _chunk_key(chunk) in packed_keys]
        for query, chunks in zip(plan.subqueries, responses, strict=True)
    }
    return AgentState(
        subqueries=list(plan.subqueries),
        plan=[*state.get("plan", []), *plan.subqueries],
        retrieved_chunks=[chunk.model_dump(mode="json") for chunk in packed_chunks],
        evidence_coverage=coverage,
        context_tokens=packed.total_tokens,
        retry_count=state.get("retry_count", 0) + 1,
        remaining_tool_calls=state.get("remaining_tool_calls", 0) - tool_calls,
        remaining_agent_rounds=state.get("remaining_agent_rounds", 0) - 1,
        response={"action": "multihop_retrieved", "count": len(packed_chunks)},
    )


async def _retrieve_one(
    query: str,
    context: AgentRuntimeContext,
    semaphore: asyncio.Semaphore,
) -> list[RetrievedChunk]:
    async with semaphore:
        return await context.retriever.aretrieve(
            query,
            context.auth_filter,
            context.limits.retrieval_top_k,
        )


async def _cancel(tasks: Sequence[asyncio.Task[list[RetrievedChunk]]]) -> None:
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


def _state_chunks(state: AgentState) -> list[RetrievedChunk] | None:
    try:
        return [RetrievedChunk.model_validate(raw) for raw in state.get("retrieved_chunks", [])]
    except ValidationError:
        return None


def _deduplicate(chunks: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
    selected: dict[tuple[str, int, str | None, str], RetrievedChunk] = {}
    order: list[tuple[str, int, str | None, str]] = []
    for chunk in chunks:
        key = _chunk_key(chunk)
        current = selected.get(key)
        if current is None:
            order.append(key)
            selected[key] = chunk
        elif (chunk.score or 0.0) > (current.score or 0.0):
            selected[key] = chunk
    return [selected[key] for key in order]


def _chunk_key(chunk: RetrievedChunk) -> tuple[str, int, str | None, str]:
    return (
        chunk.document_id,
        chunk.document_version,
        chunk.index_version,
        chunk.chunk_id,
    )


def _plan_request_content(query: str, grade: EvidenceGrade) -> str:
    return json.dumps(
        {
            "query": query,
            "missing_aspects": grade.missing_aspects,
            "grader_reason": grade.reason,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def select_after_plan(state: AgentState) -> AfterPlan:
    return (
        "grade_evidence"
        if state.get("response", {}).get("action") == "multihop_retrieved"
        else "insufficient_evidence"
    )


def _failed(
    state: AgentState,
    code: str,
    reason: str,
    *,
    tool_calls: int = 0,
    clear_evidence: bool = False,
) -> AgentState:
    update = AgentState(
        grade=EvidenceGrade(
            relevant=False,
            sufficient=False,
            confidence=1.0,
            recommended_action="stop",
            reason=reason,
        ).model_dump(mode="json"),
        response={"action": "plan_failed"},
        error_codes=append_error(state, code),
        remaining_tool_calls=max(0, state.get("remaining_tool_calls", 0) - tool_calls),
        remaining_agent_rounds=max(
            0,
            state.get("remaining_agent_rounds", 0) - 1,
        ),
    )
    if clear_evidence:
        update["retrieved_chunks"] = []
        update["evidence_coverage"] = {}
        update["context_tokens"] = 0
    return update
