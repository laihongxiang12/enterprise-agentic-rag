"""Bounded evidence grading, rewrite, and retrieval-loop tests."""

import json
from collections.abc import Sequence

from app.agents.graph import build_agent_graph
from app.agents.nodes.grade import grade_request_content
from app.agents.state import AgentLimits, AgentRuntimeContext, build_initial_state
from app.core.security import BasicPolicyEngine
from app.llm.base import LLMTransientError
from app.llm.fake import FakeLLM
from app.retrieval.fakes import FakeRetriever
from app.schemas.agent import AgentState
from app.schemas.documents import Principal
from app.schemas.retrieval import AuthFilter, RetrievedChunk


def route_payload() -> str:
    return json.dumps({"route": "knowledge", "confidence": 1.0, "reason": "requires evidence"})


def grade_payload(
    *,
    sufficient: bool,
    action: str,
    reason: str = "graded",
) -> str:
    return json.dumps(
        {
            "relevant": True,
            "sufficient": sufficient,
            "confidence": 0.9,
            "missing_aspects": [] if sufficient else ["specific limit"],
            "recommended_action": action,
            "reason": reason,
        }
    )


def rewrite_payload(query: str) -> str:
    return json.dumps({"query": query, "reason": "add missing detail"})


def grounded_answer_payload(chunk: RetrievedChunk) -> str:
    return json.dumps(
        {
            "answer": f"{chunk.content} [1]",
            "citations": [
                {
                    "document_id": chunk.document_id,
                    "document_version": chunk.document_version,
                    "chunk_id": chunk.chunk_id,
                }
            ],
        }
    )


def evidence(
    chunk_id: str,
    content: str,
    *,
    tenant_id: str = "tenant-a",
) -> RetrievedChunk:
    return RetrievedChunk(
        tenant_id=tenant_id,
        workspace_id="hr",
        document_id="leave-policy",
        document_version=1,
        chunk_id=chunk_id,
        parent_id="leave-parent",
        content=content,
        title="Leave policy",
        source_uri=f"memory://{tenant_id}/leave-policy",
        acl_groups=["hr"],
        index_version="index-active",
    )


def auth_filter() -> AuthFilter:
    return AuthFilter(
        tenant_id="tenant-a",
        workspace_id="hr",
        group_ids=frozenset({"hr"}),
        allowed_document_ids=frozenset({"leave-policy"}),
        allowed_index_versions=frozenset({"index-active"}),
        allowed_document_versions=frozenset({("leave-policy", 1, "index-active")}),
    )


class SequenceClock:
    def __init__(self, values: Sequence[int]) -> None:
        self._values = list(values)
        self._last = self._values[-1]

    def __call__(self) -> int:
        if self._values:
            self._last = self._values.pop(0)
        return self._last


def runtime_context(
    *,
    grades: Sequence[str | Exception] = (),
    rewrites: Sequence[str | Exception] = (),
    responses: dict[str, Sequence[RetrievedChunk]] | None = None,
    failures: dict[str, Exception] | None = None,
    limits: AgentLimits | None = None,
    clock: SequenceClock | None = None,
) -> tuple[AgentRuntimeContext, FakeLLM, FakeRetriever]:
    answer_scripts: list[str] = []
    for chunks in reversed(list((responses or {}).values())):
        if chunks:
            answer_scripts = [grounded_answer_payload(chunks[-1])]
            break
    llm = FakeLLM(
        schema_scripts={
            "RouteDecision": [route_payload()],
            "EvidenceGrade": grades,
            "QueryRewrite": rewrites,
            "GroundedAnswer": answer_scripts,
        }
    )
    retriever = FakeRetriever(responses=responses, failures=failures)
    context = AgentRuntimeContext(
        principal=Principal(
            user_id="alice",
            tenant_id="tenant-a",
            group_ids=["hr"],
        ),
        auth_filter=auth_filter(),
        llm=llm,
        retriever=retriever,
        policy_engine=BasicPolicyEngine(),
        limits=limits or AgentLimits(),
        now_epoch_ms=clock or SequenceClock([1_000]),
    )
    return context, llm, retriever


async def run(
    context: AgentRuntimeContext,
    query: str,
    *,
    max_retries: int = 2,
) -> AgentState:
    state = build_initial_state(
        context,
        trace_id="trace-retry",
        query=query,
        max_retries=max_retries,
    )
    return await build_agent_graph().ainvoke(state, context=context)


async def test_sufficient_on_first_retrieval() -> None:
    query = "leave limit"
    context, _, retriever = runtime_context(
        grades=[grade_payload(sufficient=True, action="answer")],
        responses={query: [evidence("first", "Ten days of leave")]},
    )

    result = await run(context, query)

    assert result["response"]["action"] == "completed"
    assert result["retry_count"] == 0
    assert len(retriever.calls) == 1


async def test_second_retrieval_succeeds_after_one_rewrite() -> None:
    query = "leave limit"
    rewritten = "annual leave exact day limit"
    context, _, retriever = runtime_context(
        grades=[
            grade_payload(sufficient=False, action="rewrite"),
            grade_payload(sufficient=True, action="answer"),
        ],
        rewrites=[rewrite_payload(rewritten)],
        responses={
            query: [evidence("weak", "Leave is available")],
            rewritten: [evidence("strong", "Annual leave is exactly ten days")],
        },
    )

    result = await run(context, query)

    assert result["response"]["action"] == "completed"
    assert result["active_query"] == rewritten
    assert result["retry_count"] == 1
    assert result["remaining_tool_calls"] == 6
    assert result["remaining_agent_rounds"] == 4
    assert [call[0] for call in retriever.calls] == [query, rewritten]


async def test_zero_retry_budget_stops_after_first_retrieval() -> None:
    query = "leave limit"
    context, _, retriever = runtime_context(
        grades=[grade_payload(sufficient=False, action="rewrite")],
        responses={query: [evidence("weak", "Leave is available")]},
    )

    result = await run(context, query, max_retries=0)

    assert result["response"]["action"] == "insufficient_evidence"
    assert result["retry_count"] == 0
    assert len(retriever.calls) == 1


async def test_always_insufficient_stops_after_two_extra_retrievals() -> None:
    query = "leave limit"
    second = "leave exact limit"
    third = "annual leave exact numeric limit"
    context, _, retriever = runtime_context(
        grades=[grade_payload(sufficient=False, action="rewrite")] * 3,
        rewrites=[rewrite_payload(second), rewrite_payload(third)],
        responses={
            query: [evidence("one", "Leave exists")],
            second: [evidence("two", "Leave has a limit")],
            third: [evidence("three", "Limit remains unclear")],
        },
    )

    result = await run(context, query, max_retries=2)

    assert result["response"] == {
        "action": "insufficient_evidence",
        "reason": "graded",
        "retries": 2,
    }
    assert len(retriever.calls) == 3
    assert result["remaining_tool_calls"] == 5


async def test_empty_evidence_refuses_without_llm_grading() -> None:
    query = "leave limit"
    context, llm, retriever = runtime_context(responses={query: []})

    result = await run(context, query)

    assert result["response"]["action"] == "insufficient_evidence"
    assert result["grade"]["reason"] == "no authorized evidence was retrieved"
    assert len(llm.calls) == 1
    assert len(retriever.calls) == 1


async def test_disabled_llm_grading_stops_conservatively() -> None:
    query = "leave limit"
    context, llm, retriever = runtime_context(
        responses={query: [evidence("first", "Ten days")]},
        limits=AgentLimits(llm_grading_enabled=False),
    )

    result = await run(context, query)

    assert result["response"]["action"] == "insufficient_evidence"
    assert result["grade"]["reason"] == "LLM grading is disabled; sufficiency is unknown"
    assert len(llm.calls) == 1
    assert len(retriever.calls) == 1


def test_grader_prompt_is_valid_json_within_character_budget() -> None:
    prompt = grade_request_content(
        "quoted query " + '"\\' * 200,
        [evidence("first", 'content "\\' * 200)],
        max_chars=256,
    )

    decoded = json.loads(prompt)
    assert len(prompt) <= 256
    assert decoded["query"]
    assert isinstance(decoded["untrusted_evidence"], list)


async def test_grader_timeout_stops_safely() -> None:
    query = "leave limit"
    context, _, retriever = runtime_context(
        grades=[LLMTransientError("timeout")],
        responses={query: [evidence("first", "Ten days")]},
    )

    result = await run(context, query)

    assert result["response"]["action"] == "insufficient_evidence"
    assert "grader_failed" in result["error_codes"]
    assert len(retriever.calls) == 1


async def test_unchanged_rewrite_stops_without_extra_retrieval() -> None:
    query = "leave limit"
    context, _, retriever = runtime_context(
        grades=[grade_payload(sufficient=False, action="rewrite")],
        rewrites=[rewrite_payload("  LEAVE   LIMIT  ")],
        responses={query: [evidence("weak", "Leave exists")]},
    )

    result = await run(context, query)

    assert result["response"]["action"] == "insufficient_evidence"
    assert "rewrite_unchanged" in result["error_codes"]
    assert result["retry_count"] == 0
    assert len(retriever.calls) == 1


async def test_deadline_during_loop_prevents_extra_retrieval() -> None:
    query = "leave limit"
    clock = SequenceClock([1_000, 1_000, 1_000, 1_000, 1_000, 32_000])
    context, _, retriever = runtime_context(
        grades=[grade_payload(sufficient=False, action="rewrite")],
        rewrites=[rewrite_payload("unused rewrite")],
        responses={query: [evidence("weak", "Leave exists")]},
        clock=clock,
    )

    result = await run(context, query)

    assert result["response"]["action"] == "insufficient_evidence"
    assert "deadline_exceeded" in result["error_codes"]
    assert len(retriever.calls) == 1


async def test_retriever_exception_becomes_controlled_insufficient_result() -> None:
    query = "leave limit"
    context, llm, retriever = runtime_context(
        failures={query: RuntimeError("database details must not leak")}
    )

    result = await run(context, query)

    assert result["response"]["action"] == "insufficient_evidence"
    assert "retrieval_failed" in result["error_codes"]
    assert "database details" not in json.dumps(result)
    assert len(llm.calls) == 1
    assert len(retriever.calls) == 1


async def test_unauthorized_retrieval_result_never_reaches_grader() -> None:
    query = "leave limit"
    context, llm, _ = runtime_context(
        responses={query: [evidence("secret", "Other tenant secret", tenant_id="tenant-b")]}
    )

    result = await run(context, query)

    assert result["response"]["action"] == "insufficient_evidence"
    assert "unauthorized_evidence" in result["error_codes"]
    assert result["retrieved_chunks"] == []
    assert len(llm.calls) == 1
