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
    plan: list[str]
    retrieved_chunks: list[dict[str, object]]
    grade: dict[str, object]
    retry_count: int
    max_retries: int
    remaining_tool_calls: int
    deadline_epoch_ms: int
    response: dict[str, object]
    error_codes: list[str]


def new_agent_state(*, trace_id: str, tenant_id: str, user_id: str) -> AgentState:
    """Create state with fresh mutable containers for every request."""
    return AgentState(
        trace_id=trace_id,
        tenant_id=tenant_id,
        user_id=user_id,
        plan=[],
        retrieved_chunks=[],
        grade={},
        response={},
        error_codes=[],
        retry_count=0,
    )
