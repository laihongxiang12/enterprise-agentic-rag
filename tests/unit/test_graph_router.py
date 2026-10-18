"""Stage 11 bounded router graph and server-side policy tests."""

import json

from app.agents.graph import build_agent_graph
from app.agents.state import AgentLimits, AgentRuntimeContext, build_initial_state
from app.core.security import BasicPolicyEngine
from app.llm.fake import FakeLLM
from app.retrieval.fakes import FakeRetriever
from app.schemas.agent import AgentState
from app.schemas.documents import Principal
from app.schemas.retrieval import AuthFilter, RetrievedChunk


def route_payload(route: str, *, confidence: float = 0.9) -> str:
    return json.dumps({"route": route, "confidence": confidence, "reason": f"choose {route}"})


def chunk() -> RetrievedChunk:
    return RetrievedChunk(
        tenant_id="tenant-a",
        workspace_id="hr",
        document_id="leave-policy",
        document_version=1,
        chunk_id="leave-1",
        parent_id="leave-parent",
        content="员工年假为十天。",
        title="年假制度",
        source_uri="memory://tenant-a/leave-policy",
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


def context(
    query: str,
    route: str,
    *,
    confidence: float = 0.9,
    scopes: tuple[str, ...] = (),
    limits: AgentLimits | None = None,
    response_override: str | None = None,
) -> tuple[AgentRuntimeContext, FakeLLM, FakeRetriever]:
    llm = FakeLLM(
        responses={query: response_override or route_payload(route, confidence=confidence)}
    )
    retriever = FakeRetriever(responses={query: [chunk()]})
    runtime_context = AgentRuntimeContext(
        principal=Principal(
            user_id="alice",
            tenant_id="tenant-a",
            group_ids=["hr"],
            scopes=list(scopes),
        ),
        auth_filter=auth_filter(),
        llm=llm,
        retriever=retriever,
        policy_engine=BasicPolicyEngine(),
        limits=limits or AgentLimits(),
        now_epoch_ms=lambda: 1_000,
    )
    return runtime_context, llm, retriever


async def run_graph(runtime_context: AgentRuntimeContext, query: str) -> AgentState:
    graph = build_agent_graph()
    state = build_initial_state(runtime_context, trace_id="trace-1", query=query)
    return await graph.ainvoke(state, context=runtime_context)


async def test_direct_route_finishes_without_retrieval() -> None:
    runtime_context, llm, retriever = context("Explain recursion", "direct")

    result = await run_graph(runtime_context, "Explain recursion")

    assert result["route"] == "direct"
    assert result["route_reason"] == "choose direct"
    assert result["response"]["action"] == "direct"
    assert result["remaining_agent_rounds"] == 5
    assert len(llm.calls) == 1
    assert retriever.calls == []


async def test_knowledge_route_uses_exact_trusted_filter() -> None:
    query = "What does the handbook say?"
    runtime_context, _, retriever = context(query, "knowledge")

    result = await run_graph(runtime_context, query)

    assert result["response"] == {"action": "retrieved", "count": 1}
    assert result["retrieved_chunks"][0]["chunk_id"] == "leave-1"
    assert result["remaining_tool_calls"] == 7
    assert retriever.calls == [(query, runtime_context.auth_filter, 20)]


async def test_unknown_schema_fails_safe_to_knowledge() -> None:
    query = "ambiguous request"
    invalid = route_payload("unknown")
    runtime_context, _, retriever = context(
        query,
        "direct",
        response_override=invalid,
    )

    result = await run_graph(runtime_context, query)

    assert result["route"] == "knowledge"
    assert result["route_confidence"] == 0.0
    assert "router_fallback" in result["error_codes"]
    assert len(retriever.calls) == 1


async def test_internal_fact_cannot_use_model_selected_direct_route() -> None:
    query = "公司的年假制度是什么?"
    runtime_context, _, retriever = context(query, "direct")

    result = await run_graph(runtime_context, query)

    assert result["route"] == "knowledge"
    assert "overridden" in result["route_reason"]
    assert len(retriever.calls) == 1


async def test_low_confidence_direct_route_escalates_to_retrieval() -> None:
    query = "Could this require a source?"
    runtime_context, _, retriever = context(query, "direct", confidence=0.2)

    result = await run_graph(runtime_context, query)

    assert result["route"] == "knowledge"
    assert len(retriever.calls) == 1


async def test_sql_is_blocked_when_feature_flag_is_off_even_with_scope() -> None:
    query = "query the finance table"
    runtime_context, _, retriever = context(
        query,
        "sql",
        scopes=("tools:sql",),
    )

    result = await run_graph(runtime_context, query)

    assert result["route"] == "policy_blocked"
    assert result["response"]["action"] == "policy_blocked"
    assert "sql_disabled" in result["error_codes"]
    assert retriever.calls == []


async def test_web_is_blocked_when_enabled_but_principal_is_unauthorized() -> None:
    query = "search the public web"
    runtime_context, _, retriever = context(
        query,
        "web",
        limits=AgentLimits(web_enabled=True),
    )

    result = await run_graph(runtime_context, query)

    assert result["route"] == "policy_blocked"
    assert "web_unauthorized" in result["error_codes"]
    assert retriever.calls == []


async def test_identity_scope_mismatch_is_blocked_before_retrieval() -> None:
    query = "handbook source"
    runtime_context, _, retriever = context(query, "knowledge")
    state = build_initial_state(runtime_context, trace_id="trace-1", query=query)
    state["tenant_id"] = "tenant-b"

    result = await build_agent_graph().ainvoke(state, context=runtime_context)

    assert result["route"] == "policy_blocked"
    assert "identity_scope_mismatch" in result["error_codes"]
    assert retriever.calls == []


async def test_incomplete_authorization_scope_is_blocked_before_retrieval() -> None:
    query = "handbook source"
    runtime_context, _, retriever = context(query, "knowledge")
    incomplete_context = AgentRuntimeContext(
        principal=runtime_context.principal,
        auth_filter=AuthFilter(tenant_id="tenant-a", workspace_id="hr"),
        llm=runtime_context.llm,
        retriever=retriever,
        policy_engine=runtime_context.policy_engine,
        limits=runtime_context.limits,
        now_epoch_ms=runtime_context.now_epoch_ms,
    )

    result = await run_graph(incomplete_context, query)

    assert result["route"] == "policy_blocked"
    assert "empty_authorization_scope" in result["error_codes"]
    assert retriever.calls == []


async def test_expired_deadline_stops_before_llm_or_retrieval() -> None:
    query = "handbook source"
    runtime_context, llm, retriever = context(query, "knowledge")
    state = build_initial_state(runtime_context, trace_id="trace-1", query=query)
    state["deadline_epoch_ms"] = 999

    result = await build_agent_graph().ainvoke(state, context=runtime_context)

    assert result["route"] == "policy_blocked"
    assert "deadline_exceeded" in result["error_codes"]
    assert llm.calls == []
    assert retriever.calls == []


async def test_round_budget_stops_before_llm() -> None:
    query = "handbook source"
    runtime_context, llm, _ = context(query, "knowledge")
    state = build_initial_state(runtime_context, trace_id="trace-1", query=query)
    state["remaining_agent_rounds"] = 0

    result = await build_agent_graph().ainvoke(state, context=runtime_context)

    assert result["route"] == "policy_blocked"
    assert "round_budget_exhausted" in result["error_codes"]
    assert llm.calls == []


async def test_tool_budget_blocks_knowledge_route_before_retrieval() -> None:
    query = "handbook source"
    runtime_context, _, retriever = context(query, "knowledge")
    state = build_initial_state(runtime_context, trace_id="trace-1", query=query)
    state["remaining_tool_calls"] = 0

    result = await build_agent_graph().ainvoke(state, context=runtime_context)

    assert result["route"] == "policy_blocked"
    assert "tool_budget_exhausted" in result["error_codes"]
    assert retriever.calls == []
