"""Bounded multi-hop planning, retrieval, authorization, and context tests."""

import asyncio
import json
from collections.abc import Sequence

import pytest
from pydantic import ValidationError

from app.agents.graph import build_agent_graph
from app.agents.state import AgentLimits, AgentRuntimeContext, build_initial_state
from app.core.security import BasicPolicyEngine
from app.llm.fake import FakeLLM
from app.retrieval.base import Retriever
from app.retrieval.context import ContextItem, ContextPacker
from app.retrieval.fakes import FakeRetriever
from app.schemas.agent import AgentState, MultiHopPlan
from app.schemas.documents import Principal
from app.schemas.retrieval import AuthFilter, RetrievedChunk


def route_payload() -> str:
    return json.dumps({"route": "knowledge", "confidence": 1.0, "reason": "requires evidence"})


def grade_payload(*, action: str, sufficient: bool = False) -> str:
    return json.dumps(
        {
            "relevant": True,
            "sufficient": sufficient,
            "confidence": 0.9,
            "missing_aspects": [] if sufficient else ["two policy sections"],
            "recommended_action": action,
            "reason": "combine evidence" if not sufficient else "complete",
        }
    )


def plan_payload(*queries: str) -> str:
    return json.dumps({"subqueries": list(queries), "reason": "multi-hop"})


def grounded_answer_payload(item: RetrievedChunk) -> str:
    return json.dumps(
        {
            "answer": f"{item.content} [1]",
            "citations": [
                {
                    "document_id": item.document_id,
                    "document_version": item.document_version,
                    "chunk_id": item.chunk_id,
                }
            ],
        }
    )


def chunk(
    chunk_id: str,
    content: str,
    *,
    document_id: str = "policy",
    source_uri: str | None = None,
    tenant_id: str = "tenant-a",
    score: float = 0.8,
) -> RetrievedChunk:
    return RetrievedChunk(
        tenant_id=tenant_id,
        workspace_id="ops",
        document_id=document_id,
        document_version=1,
        chunk_id=chunk_id,
        parent_id=f"parent-{chunk_id}",
        content=content,
        title="Operations policy",
        source_uri=source_uri or f"memory://tenant-a/{document_id}",
        acl_groups=["ops"],
        index_version="index-active",
        score=score,
    )


def auth_filter() -> AuthFilter:
    documents = {"policy", "reporting", "timeline", "device", "security"}
    return AuthFilter(
        tenant_id="tenant-a",
        workspace_id="ops",
        group_ids=frozenset({"ops"}),
        allowed_document_ids=frozenset(documents),
        allowed_index_versions=frozenset({"index-active"}),
        allowed_document_versions=frozenset(
            (document_id, 1, "index-active") for document_id in documents
        ),
    )


class FixedClock:
    def __init__(self, value: int = 1_000) -> None:
        self.value = value

    def __call__(self) -> int:
        return self.value


def context(
    *,
    retriever: Retriever,
    plans: Sequence[str],
    grades: Sequence[str],
    answers: Sequence[str | Exception] = (),
    limits: AgentLimits | None = None,
    clock: FixedClock | None = None,
) -> tuple[AgentRuntimeContext, FakeLLM]:
    llm = FakeLLM(
        schema_scripts={
            "RouteDecision": [route_payload()],
            "EvidenceGrade": grades,
            "MultiHopPlan": plans,
            "GroundedAnswer": answers,
        }
    )
    return (
        AgentRuntimeContext(
            principal=Principal(
                user_id="alice",
                tenant_id="tenant-a",
                group_ids=["ops"],
            ),
            auth_filter=auth_filter(),
            llm=llm,
            retriever=retriever,
            policy_engine=BasicPolicyEngine(),
            limits=limits or AgentLimits(),
            now_epoch_ms=clock or FixedClock(),
        ),
        llm,
    )


async def run(
    runtime_context: AgentRuntimeContext, query: str = "lost laptop process"
) -> AgentState:
    state = build_initial_state(
        runtime_context,
        trace_id="trace-multihop",
        query=query,
    )
    return await build_agent_graph().ainvoke(state, context=runtime_context)


async def test_two_hop_plan_combines_evidence_and_reuses_trusted_filter() -> None:
    query = "lost laptop process"
    first = "how to report a lost laptop"
    second = "lost laptop reporting deadline"
    retriever = FakeRetriever(
        responses={
            query: [chunk("weak", "A reporting process exists")],
            first: [chunk("report", "Report to security", document_id="reporting")],
            second: [chunk("time", "Report within one hour", document_id="timeline")],
        }
    )
    runtime_context, _ = context(
        retriever=retriever,
        plans=[plan_payload(first, second)],
        grades=[grade_payload(action="decompose"), grade_payload(action="answer", sufficient=True)],
        answers=[
            grounded_answer_payload(chunk("report", "Report to security", document_id="reporting"))
        ],
    )

    result = await run(runtime_context, query)

    assert result["response"]["action"] == "completed"
    assert result["subqueries"] == [first, second]
    assert result["retry_count"] == 1
    assert result["remaining_tool_calls"] == 5
    assert result["evidence_coverage"] == {first: ["report"], second: ["time"]}
    assert [call[0] for call in retriever.calls] == [query, first, second]
    assert all(call[1] == runtime_context.auth_filter for call in retriever.calls)


async def test_four_subqueries_are_the_hard_fanout_limit() -> None:
    query = "combine four sections"
    subqueries = tuple(f"section {number}" for number in range(4))
    retriever = FakeRetriever(
        responses={
            query: [chunk("weak", "incomplete")],
            **{
                subquery: [chunk(f"hit-{number}", f"answer {number}")]
                for number, subquery in enumerate(subqueries)
            },
        }
    )
    runtime_context, _ = context(
        retriever=retriever,
        plans=[plan_payload(*subqueries)],
        grades=[grade_payload(action="decompose"), grade_payload(action="answer", sufficient=True)],
        answers=[grounded_answer_payload(chunk("weak", "incomplete"))],
    )

    result = await run(runtime_context, query)

    assert result["response"]["action"] == "completed"
    assert len(result["subqueries"]) == 4
    assert len(retriever.calls) == 5
    assert result["remaining_tool_calls"] == 3


def test_plan_schema_rejects_five_or_duplicate_subqueries() -> None:
    with pytest.raises(ValidationError):
        MultiHopPlan(subqueries=[f"q{number}" for number in range(5)], reason="too many")
    with pytest.raises(ValidationError):
        MultiHopPlan(subqueries=["same query", " SAME   QUERY "], reason="duplicate")


async def test_duplicate_hits_are_merged_once_with_coverage_preserved() -> None:
    query = "combined policy"
    first = "part one"
    second = "part two"
    shared = chunk("shared", "shared evidence", score=0.9)
    retriever = FakeRetriever(responses={query: [shared], first: [shared], second: [shared]})
    runtime_context, _ = context(
        retriever=retriever,
        plans=[plan_payload(first, second)],
        grades=[grade_payload(action="decompose"), grade_payload(action="answer", sufficient=True)],
        answers=[grounded_answer_payload(shared)],
    )

    result = await run(runtime_context, query)

    assert [item["chunk_id"] for item in result["retrieved_chunks"]] == ["shared"]
    assert result["evidence_coverage"] == {first: ["shared"], second: ["shared"]}


class BlockingRetriever:
    def __init__(self, initial_query: str) -> None:
        self.initial_query = initial_query
        self.calls: list[tuple[str, AuthFilter, int]] = []
        self.cancelled = 0
        self.active = 0
        self.max_active = 0

    async def aretrieve(
        self, query: str, trusted_filter: AuthFilter, top_k: int
    ) -> list[RetrievedChunk]:
        self.calls.append((query, trusted_filter, top_k))
        if query == self.initial_query:
            return [chunk("weak", "incomplete")]
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        finally:
            self.active -= 1
        return []


async def test_timeout_cancels_fanout_and_respects_semaphore() -> None:
    query = "slow multi hop"
    retriever = BlockingRetriever(query)
    runtime_context, _ = context(
        retriever=retriever,
        plans=[plan_payload("one", "two", "three", "four")],
        grades=[grade_payload(action="decompose")],
        limits=AgentLimits(deadline_seconds=0.03, subquery_concurrency=2),
    )

    result = await run(runtime_context, query)

    assert result["response"]["action"] == "insufficient_evidence"
    assert "multihop_timeout" in result["error_codes"]
    assert retriever.max_active == 2
    assert retriever.cancelled == 2
    assert result["remaining_tool_calls"] == 3


async def test_unauthorized_subquery_evidence_is_cleared_before_second_grade() -> None:
    query = "cross tenant request"
    first = "allowed section"
    second = "other tenant secret"
    retriever = FakeRetriever(
        responses={
            query: [chunk("weak", "incomplete")],
            first: [chunk("allowed", "allowed")],
            second: [chunk("secret", "secret", tenant_id="tenant-b")],
        }
    )
    runtime_context, llm = context(
        retriever=retriever,
        plans=[plan_payload(first, second)],
        grades=[grade_payload(action="decompose")],
    )

    result = await run(runtime_context, query)

    assert result["response"]["action"] == "insufficient_evidence"
    assert "unauthorized_evidence" in result["error_codes"]
    assert result["retrieved_chunks"] == []
    assert len(llm.calls) == 3


def test_context_packer_prioritizes_source_diversity_and_token_budget() -> None:
    candidates = [
        ContextItem(
            child=chunk("a1", "a b c d", source_uri="memory://source-a", score=0.9),
            content="a b c d",
            used_parent=True,
            truncated=False,
            token_count=4,
        ),
        ContextItem(
            child=chunk("a2", "e f g", source_uri="memory://source-a", score=0.8),
            content="e f g",
            used_parent=True,
            truncated=False,
            token_count=3,
        ),
        ContextItem(
            child=chunk("b1", "h i j k", source_uri="memory://source-b", score=0.7),
            content="h i j k",
            used_parent=True,
            truncated=False,
            token_count=4,
        ),
    ]

    packed = ContextPacker(token_budget=6).pack(candidates)

    assert [item.child.chunk_id for item in packed.items] == ["a1", "b1"]
    assert packed.items[1].truncated
    assert packed.total_tokens == 6
    assert packed.total_tokens <= packed.token_budget


async def test_deadline_after_fanout_stops_before_second_grade() -> None:
    query = "deadline query"
    retriever = FakeRetriever(
        responses={
            query: [chunk("weak", "incomplete")],
            "one": [chunk("one", "first")],
            "two": [chunk("two", "second")],
        }
    )
    clock = FixedClock()
    runtime_context, llm = context(
        retriever=retriever,
        plans=[plan_payload("one", "two")],
        grades=[grade_payload(action="decompose")],
        clock=clock,
    )
    original_retrieve = retriever.aretrieve

    async def expire_after_retrieve(
        subquery: str, trusted_filter: AuthFilter, top_k: int
    ) -> list[RetrievedChunk]:
        result = await original_retrieve(subquery, trusted_filter, top_k)
        if subquery in {"one", "two"}:
            clock.value = 32_000
        return result

    retriever.aretrieve = expire_after_retrieve  # type: ignore[method-assign]

    result = await run(runtime_context, query)

    assert result["response"]["action"] == "insufficient_evidence"
    assert "deadline_exceeded" in result["error_codes"]
    assert result["retrieved_chunks"] == []
    assert len(llm.calls) == 3
