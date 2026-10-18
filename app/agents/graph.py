"""Minimal bounded LangGraph: analyze -> policy -> terminal action."""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.nodes.analyze import analyze_and_route
from app.agents.nodes.retrieve import direct_response, policy_blocked, retrieve
from app.agents.nodes.route import policy_check, select_next_node
from app.agents.state import AgentRuntimeContext
from app.schemas.agent import AgentState


def build_agent_graph() -> CompiledStateGraph[
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
    builder.add_edge("direct", END)
    builder.add_edge("retrieve", END)
    builder.add_edge("policy_blocked", END)
    return builder.compile(name="enterprise-agent-router")
