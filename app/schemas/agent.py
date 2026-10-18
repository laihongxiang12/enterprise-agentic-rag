"""Agent routing, grading, and state contracts."""

from typing import Literal, TypedDict

from pydantic import Field

from app.schemas.base import ContractModel


class RouteDecision(ContractModel):
    route: Literal["knowledge", "direct", "sql", "web"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class EvidenceGrade(ContractModel):
    relevant: bool
    sufficient: bool
    confidence: float = Field(ge=0.0, le=1.0)
    missing_aspects: list[str] = Field(default_factory=list)
    recommended_action: Literal["answer", "rewrite", "decompose", "stop"]
    reason: str


class AgentState(TypedDict, total=False):
    trace_id: str
    tenant_id: str
    user_id: str
    session_id: str
    workspace_id: str | None
    original_query: str
    active_query: str
    route: str
    route_confidence: float
    route_reason: str
    plan: list[str]
    retrieved_chunks: list[dict[str, object]]
    grade: dict[str, object]
    retry_count: int
    max_retries: int
    remaining_tool_calls: int
    remaining_agent_rounds: int
    deadline_epoch_ms: int
    response: dict[str, object]
    error_codes: list[str]


def new_agent_state(
    *,
    trace_id: str,
    tenant_id: str,
    user_id: str,
    original_query: str = "",
    workspace_id: str | None = None,
    max_retries: int = 2,
    remaining_tool_calls: int = 8,
    remaining_agent_rounds: int = 6,
    deadline_epoch_ms: int = 0,
) -> AgentState:
    """Create state with fresh mutable containers for every request."""
    return AgentState(
        trace_id=trace_id,
        tenant_id=tenant_id,
        user_id=user_id,
        workspace_id=workspace_id,
        original_query=original_query,
        active_query=original_query,
        plan=[],
        retrieved_chunks=[],
        grade={},
        response={},
        error_codes=[],
        retry_count=0,
        max_retries=max_retries,
        remaining_tool_calls=remaining_tool_calls,
        remaining_agent_rounds=remaining_agent_rounds,
        deadline_epoch_ms=deadline_epoch_ms,
    )
