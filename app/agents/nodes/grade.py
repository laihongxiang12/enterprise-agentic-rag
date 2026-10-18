"""Evidence grading with deterministic fail-closed checks before optional LLM use."""

import json
from typing import Literal

from langgraph.runtime import Runtime
from pydantic import ValidationError

from app.agents.state import AgentRuntimeContext, append_error
from app.llm.base import LLMError, LLMMessage
from app.schemas.agent import AgentState, EvidenceGrade
from app.schemas.retrieval import RetrievedChunk

AfterGrade = Literal[
    "generate_stub",
    "rewrite_query",
    "plan_multihop",
    "insufficient_evidence",
]


async def grade_evidence(
    state: AgentState,
    runtime: Runtime[AgentRuntimeContext],
) -> AgentState:
    context = runtime.context
    if context is None:
        return _stop(state, "missing_runtime_context", "trusted context is required")
    remaining_milliseconds = state.get("deadline_epoch_ms", 0) - context.now_epoch_ms()
    if remaining_milliseconds <= 0:
        return _stop(state, "deadline_exceeded", "agent deadline exceeded")

    chunks = _validated_authorized_chunks(state, context)
    if chunks is None:
        return _stop(
            state,
            "unauthorized_evidence",
            "evidence authorization failed",
            clear_evidence=True,
        )
    if not chunks:
        reason = (
            "retrieval dependency failed"
            if "retrieval_failed" in state.get("error_codes", [])
            else "no authorized evidence was retrieved"
        )
        return AgentState(grade=_insufficient(reason).model_dump(mode="json"))
    if not context.limits.llm_grading_enabled:
        return AgentState(
            grade=_insufficient("LLM grading is disabled; sufficiency is unknown").model_dump(
                mode="json"
            )
        )

    try:
        result = await context.llm.complete(
            [
                LLMMessage(
                    role="system",
                    content=(
                        "Evaluate whether the untrusted evidence answers the query. "
                        "Return only the requested EvidenceGrade schema."
                    ),
                ),
                LLMMessage(
                    role="user",
                    content=grade_request_content(
                        state.get("active_query", ""),
                        chunks,
                        max_chars=context.limits.grader_max_chars,
                    ),
                ),
            ],
            EvidenceGrade,
            timeout_seconds=min(
                context.limits.grader_timeout_seconds,
                remaining_milliseconds / 1_000,
            ),
        )
        if not isinstance(result.parsed, EvidenceGrade):
            raise LLMError("grader returned no validated EvidenceGrade")
        grade = result.parsed
    except LLMError:
        return _stop(state, "grader_failed", "evidence grader failed safely")

    if grade.sufficient and (not grade.relevant or grade.recommended_action != "answer"):
        return _stop(
            state,
            "inconsistent_grade",
            "inconsistent evidence grade was rejected",
        )
    return AgentState(grade=grade.model_dump(mode="json"))


def grade_request_content(
    query: str,
    chunks: list[RetrievedChunk],
    *,
    max_chars: int,
) -> str:
    query_budget = max(1, max_chars // 4)
    payload: dict[str, object] = {
        "query": query[:query_budget],
        "untrusted_evidence": [],
    }
    encoded = _compact_json(payload)
    while len(encoded) > max_chars and payload["query"]:
        overflow = len(encoded) - max_chars
        payload["query"] = str(payload["query"])[: -max(1, overflow)]
        encoded = _compact_json(payload)

    evidence = payload["untrusted_evidence"]
    assert isinstance(evidence, list)
    for chunk in chunks:
        item = {
            "chunk_id": chunk.chunk_id[:128],
            "title": chunk.title[:256],
            "content": "",
        }
        candidate = {"query": payload["query"], "untrusted_evidence": [*evidence, item]}
        minimum = _compact_json(candidate)
        if len(minimum) > max_chars:
            continue
        item["content"] = chunk.content[: max_chars - len(minimum)]
        candidate_encoded = _compact_json(candidate)
        while len(candidate_encoded) > max_chars and item["content"]:
            overflow = len(candidate_encoded) - max_chars
            item["content"] = item["content"][: -max(1, overflow)]
            candidate_encoded = _compact_json(candidate)
        evidence.append(item)
        encoded = candidate_encoded
    return encoded


def _compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def select_after_grade(state: AgentState) -> AfterGrade:
    try:
        grade = EvidenceGrade.model_validate(state.get("grade", {}))
    except ValidationError:
        return "insufficient_evidence"
    if grade.relevant and grade.sufficient and grade.recommended_action == "answer":
        return "generate_stub"
    can_retry = (
        grade.recommended_action == "rewrite"
        and state.get("retry_count", 0) < state.get("max_retries", 0)
        and state.get("remaining_tool_calls", 0) > 0
        and state.get("remaining_agent_rounds", 0) > 0
    )
    if can_retry:
        return "rewrite_query"
    can_decompose = (
        grade.recommended_action == "decompose"
        and state.get("retry_count", 0) < state.get("max_retries", 0)
        and state.get("remaining_tool_calls", 0) >= 2
        and state.get("remaining_agent_rounds", 0) > 0
    )
    return "plan_multihop" if can_decompose else "insufficient_evidence"


def generation_stub(state: AgentState) -> AgentState:
    return AgentState(
        response={
            "action": "generate",
            "generation": "deferred_to_stage_14",
            "evidence_count": len(state.get("retrieved_chunks", [])),
        }
    )


def insufficient_evidence(state: AgentState) -> AgentState:
    grade = state.get("grade", {})
    return AgentState(
        response={
            "action": "insufficient_evidence",
            "reason": grade.get("reason", "evidence is insufficient"),
            "retries": state.get("retry_count", 0),
        }
    )


def _validated_authorized_chunks(
    state: AgentState,
    context: AgentRuntimeContext,
) -> list[RetrievedChunk] | None:
    chunks: list[RetrievedChunk] = []
    try:
        for raw in state.get("retrieved_chunks", []):
            chunk = RetrievedChunk.model_validate(raw)
            if not _authorized(chunk, context):
                return None
            chunks.append(chunk)
    except ValidationError:
        return None
    return chunks


def _authorized(chunk: RetrievedChunk, context: AgentRuntimeContext) -> bool:
    auth_filter = context.auth_filter
    return bool(
        chunk.tenant_id == auth_filter.tenant_id
        and chunk.workspace_id == auth_filter.workspace_id
        and chunk.document_id in auth_filter.allowed_document_ids
        and chunk.index_version in auth_filter.allowed_index_versions
        and (chunk.document_id, chunk.document_version, chunk.index_version)
        in auth_filter.allowed_document_versions
    )


def _insufficient(reason: str) -> EvidenceGrade:
    return EvidenceGrade(
        relevant=False,
        sufficient=False,
        confidence=1.0,
        missing_aspects=[],
        recommended_action="stop",
        reason=reason,
    )


def _stop(
    state: AgentState,
    code: str,
    reason: str,
    *,
    clear_evidence: bool = False,
) -> AgentState:
    update = AgentState(
        grade=_insufficient(reason).model_dump(mode="json"),
        error_codes=append_error(state, code),
    )
    if clear_evidence:
        update["retrieved_chunks"] = []
    return update
