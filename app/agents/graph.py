"""Bounded route, retrieval, evidence grading, and rewrite graph."""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.nodes.analyze import analyze_and_route
from app.agents.nodes.generate import generate_grounded, select_after_generate
from app.agents.nodes.grade import (
    grade_evidence,
    insufficient_evidence,
    select_after_grade,
)
from app.agents.nodes.plan import plan_and_retrieve, select_after_plan
from app.agents.nodes.retrieve import direct_response, policy_blocked, retrieve
from app.agents.nodes.rewrite import rewrite_query, select_after_rewrite
from app.agents.nodes.route import policy_check, select_next_node
from app.agents.nodes.verify import select_after_verify, verify_citations
from app.agents.state import AgentRuntimeContext
from app.schemas.agent import AgentState


def build_agent_graph(
    checkpointer: BaseCheckpointSaver[str] | None = None,
) -> CompiledStateGraph[
    AgentState,
    AgentRuntimeContext,
    AgentState,
    AgentState,
]:
    builder = StateGraph(AgentState, context_schema=AgentRuntimeContext)
    builder.add_node("analyze_and_route", analyze_and_route)
    builder.add_node("policy_check", policy_check)
    builder.add_node("direct", direct_response)
    builder.add_node("retrieve", retrieve)
    builder.add_node("grade_evidence", grade_evidence)
    builder.add_node("rewrite_query", rewrite_query)
    builder.add_node("plan_multihop", plan_and_retrieve)
    builder.add_node("generate_grounded", generate_grounded)
    builder.add_node("verify_citations", verify_citations)
    builder.add_node("insufficient_evidence", insufficient_evidence)
    builder.add_node("policy_blocked", policy_blocked)

    builder.add_edge(START, "analyze_and_route")
    builder.add_edge("analyze_and_route", "policy_check")
    builder.add_conditional_edges(
        "policy_check",
        select_next_node,
        {
            "direct": "direct",
            "retrieve": "retrieve",
            "policy_blocked": "policy_blocked",
        },
    )
    builder.add_edge("retrieve", "grade_evidence")
    builder.add_conditional_edges(
        "grade_evidence",
        select_after_grade,
        {
            "generate_grounded": "generate_grounded",
            "rewrite_query": "rewrite_query",
            "plan_multihop": "plan_multihop",
            "insufficient_evidence": "insufficient_evidence",
        },
    )
    builder.add_conditional_edges(
        "generate_grounded",
        select_after_generate,
        {
            "verify_citations": "verify_citations",
            "finish": END,
        },
    )
    builder.add_conditional_edges(
        "verify_citations",
        select_after_verify,
        {
            "repair": "generate_grounded",
            "finish": END,
        },
    )
    builder.add_conditional_edges(
        "plan_multihop",
        select_after_plan,
        {
            "grade_evidence": "grade_evidence",
            "insufficient_evidence": "insufficient_evidence",
        },
    )
    builder.add_conditional_edges(
        "rewrite_query",
        select_after_rewrite,
        {
            "retrieve": "retrieve",
            "insufficient_evidence": "insufficient_evidence",
        },
    )
    builder.add_edge("direct", END)
    builder.add_edge("insufficient_evidence", END)
    builder.add_edge("policy_blocked", END)
    return builder.compile(
        checkpointer=checkpointer,
        name="enterprise-agent-retrieval-loop",
    )
