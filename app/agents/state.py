"""Trusted runtime context and initial state construction for the agent graph."""

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from app.core.config import Settings
from app.core.dependencies import PolicyEngine
from app.llm.base import LLMProvider
from app.retrieval.base import Retriever
from app.schemas.agent import AgentState, new_agent_state
from app.schemas.documents import Principal
from app.schemas.retrieval import AuthFilter


def system_epoch_ms() -> int:
    return time.time_ns() // 1_000_000


@dataclass(frozen=True)
class AgentLimits:
    route_min_confidence: float = 0.6
    route_timeout_seconds: float = 10.0
    max_rounds: int = 6
    max_tool_calls: int = 8
    deadline_seconds: float = 30.0
    retrieval_top_k: int = 20
    sql_enabled: bool = False
    web_enabled: bool = False

    def __post_init__(self) -> None:
        if not 0 <= self.route_min_confidence <= 1:
            raise ValueError("route_min_confidence must be between zero and one")
        positive = (
            self.route_timeout_seconds,
            self.max_rounds,
            self.max_tool_calls,
            self.deadline_seconds,
            self.retrieval_top_k,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("agent limits must be positive")

    @classmethod
    def from_settings(cls, settings: Settings) -> "AgentLimits":
        return cls(
            route_min_confidence=settings.agent_route_min_confidence,
            route_timeout_seconds=settings.agent_route_timeout_seconds,
            max_rounds=settings.agent_max_rounds,
            max_tool_calls=settings.agent_max_tool_calls,
            deadline_seconds=settings.agent_deadline_seconds,
            retrieval_top_k=settings.agent_retrieval_top_k,
            sql_enabled=settings.sql_tool_enabled,
            web_enabled=settings.web_tool_enabled,
        )


@dataclass(frozen=True)
class AgentRuntimeContext:
    """Immutable dependencies and authorization that never come from the LLM."""

    principal: Principal
    auth_filter: AuthFilter
    llm: LLMProvider
    retriever: Retriever
    policy_engine: PolicyEngine
    limits: AgentLimits = field(default_factory=AgentLimits)
    now_epoch_ms: Callable[[], int] = system_epoch_ms


def build_initial_state(
    context: AgentRuntimeContext,
    *,
    trace_id: str,
    query: str,
    max_retries: int = 2,
) -> AgentState:
    """Create graph state only from trusted identity and authorization context."""
    now = context.now_epoch_ms()
    return new_agent_state(
        trace_id=trace_id,
        tenant_id=context.principal.tenant_id,
        user_id=context.principal.user_id,
        workspace_id=context.auth_filter.workspace_id,
        original_query=query,
        max_retries=max_retries,
        remaining_tool_calls=context.limits.max_tool_calls,
        remaining_agent_rounds=context.limits.max_rounds,
        deadline_epoch_ms=now + int(context.limits.deadline_seconds * 1_000),
    )


def append_error(state: AgentState, code: str) -> list[str]:
    return [*state.get("error_codes", []), code]


def blocked_update(state: AgentState, code: str, reason: str) -> AgentState:
    return AgentState(
        route="policy_blocked",
        route_confidence=0.0,
        route_reason=reason,
        error_codes=append_error(state, code),
    )
