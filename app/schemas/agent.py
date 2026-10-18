"""Agent routing, grading, and state contracts."""

from typing import Literal, TypedDict

from pydantic import Field, field_validator

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


class QueryRewrite(ContractModel):
    query: str = Field(min_length=1, max_length=8000)
    reason: str


class MultiHopPlan(ContractModel):
    subqueries: list[str] = Field(min_length=2, max_length=4)
    reason: str

    @field_validator("subqueries")
    @classmethod
    def validate_unique_subqueries(cls, values: list[str]) -> list[str]:
        stripped = [value.strip() for value in values]
        if any(not value for value in stripped):
            raise ValueError("subqueries must not be blank")
        normalized = [" ".join(value.casefold().split()) for value in stripped]
        if len(set(normalized)) != len(normalized):
            raise ValueError("subqueries must be unique")
        return stripped


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
    subqueries: list[str]
    retrieved_chunks: list[dict[str, object]]
    evidence_coverage: dict[str, list[str]]
    context_tokens: int
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
        subqueries=[],
        retrieved_chunks=[],
        evidence_coverage={},
        context_tokens=0,
        grade={},
        response={},
        error_codes=[],
        retry_count=0,
        max_retries=max_retries,
        remaining_tool_calls=remaining_tool_calls,
        remaining_agent_rounds=remaining_agent_rounds,
        deadline_epoch_ms=deadline_epoch_ms,
    )
