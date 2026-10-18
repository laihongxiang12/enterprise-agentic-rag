"""Deterministic citation verification with at most one bounded repair."""

from __future__ import annotations

import re
from typing import Literal

from langgraph.runtime import Runtime
from pydantic import ValidationError

from app.agents.nodes.generate import REFUSAL_ANSWER
from app.agents.nodes.retrieve import is_authorized_chunk
from app.agents.state import AgentRuntimeContext, append_error
from app.retrieval.citations import CitationBuilder, CitationBuildError, CitationRequest
from app.retrieval.context import ContextItem, ExpandedContext, RegexTokenCounter
from app.schemas.agent import AgentState, GroundedAnswer
from app.schemas.retrieval import RetrievedChunk

AfterVerify = Literal["repair", "finish"]
_MARKER = re.compile(r"\[(\d+)]")


async def verify_citations(
    state: AgentState,
    runtime: Runtime[AgentRuntimeContext],
) -> AgentState:
    context = runtime.context
    if context is None:
        return _refuse(state, "missing_runtime_context")
    try:
        draft = GroundedAnswer.model_validate(state.get("draft_answer", {}))
        chunks = [RetrievedChunk.model_validate(raw) for raw in state.get("retrieved_chunks", [])]
    except ValidationError:
        return _refuse(state, "invalid_citation_state")
    if not chunks or any(not is_authorized_chunk(chunk, context) for chunk in chunks):
        return _refuse(
            state,
            "unauthorized_evidence",
            clear_evidence=True,
        )

    try:
        citations = CitationBuilder().build(
            _as_context(chunks),
            context.auth_filter,
            requests=[
                CitationRequest(
                    document_id=item.document_id,
                    document_version=item.document_version,
                    chunk_id=item.chunk_id,
                )
                for item in draft.citations
            ],
        )
        expected_markers = {str(number) for number in range(1, len(citations) + 1)}
        actual_markers = set(_MARKER.findall(draft.answer))
        if actual_markers != expected_markers:
            raise CitationBuildError("answer citation markers do not match references")
    except CitationBuildError:
        return _repair_or_refuse(state, context)

    return AgentState(
        response={
            "action": "completed",
            "answer": draft.answer,
            "citations": [citation.model_dump(mode="json") for citation in citations],
            "finish_reason": "completed",
        }
    )


def select_after_verify(state: AgentState) -> AfterVerify:
    return "repair" if state.get("response", {}).get("action") == "repair_citations" else "finish"


def _as_context(chunks: list[RetrievedChunk]) -> ExpandedContext:
    counter = RegexTokenCounter()
    items = [
        ContextItem(
            child=chunk,
            content=chunk.content,
            used_parent=False,
            truncated=False,
            token_count=max(1, counter.count(chunk.content)),
        )
        for chunk in chunks
    ]
    total = sum(item.token_count for item in items)
    return ExpandedContext(
        items=items,
        total_tokens=total,
        token_budget=max(1, total),
    )


def _repair_or_refuse(
    state: AgentState,
    context: AgentRuntimeContext,
) -> AgentState:
    can_repair = (
        state.get("citation_repair_count", 0) < 1
        and state.get("remaining_agent_rounds", 0) > 0
        and state.get("deadline_epoch_ms", 0) > context.now_epoch_ms()
    )
    if not can_repair:
        return _refuse(state, "citation_verification_failed")
    return AgentState(
        citation_repair_count=state.get("citation_repair_count", 0) + 1,
        remaining_agent_rounds=state.get("remaining_agent_rounds", 0) - 1,
        response={"action": "repair_citations"},
        error_codes=append_error(state, "citation_invalid"),
    )


def _refuse(
    state: AgentState,
    code: str,
    *,
    clear_evidence: bool = False,
) -> AgentState:
    update = AgentState(
        response={
            "action": "insufficient_evidence",
            "answer": REFUSAL_ANSWER,
            "citations": [],
            "finish_reason": "insufficient_evidence",
        },
        error_codes=append_error(state, code),
    )
    if clear_evidence:
        update["retrieved_chunks"] = []
    return update
