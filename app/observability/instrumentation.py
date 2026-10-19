"""Protocol-preserving wrappers for runtime dependencies and LangGraph nodes."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import replace
from functools import wraps

from langgraph.runtime import Runtime
from pydantic import BaseModel

from app.agents.state import AgentRuntimeContext
from app.core.dependencies import PolicyEngine
from app.llm.base import LLMMessage, LLMProvider, LLMResult
from app.observability.tracing import Observability, current_trace_id
from app.retrieval.base import Reranker, Retriever
from app.schemas.agent import AgentState
from app.schemas.documents import DocumentMeta, Principal
from app.schemas.retrieval import AuthFilter, RetrievedChunk

AsyncNode = Callable[[AgentState, Runtime[AgentRuntimeContext]], Awaitable[AgentState]]
SyncRuntimeNode = Callable[[AgentState, Runtime[AgentRuntimeContext]], AgentState]
SyncStateNode = Callable[[AgentState], AgentState]
ObservedNode = AsyncNode | SyncRuntimeNode


class ObservedLLM:
    def __init__(self, backend: LLMProvider, observability: Observability) -> None:
        self._backend = backend
        self._observability = observability

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        response_schema: type[BaseModel] | None = None,
        timeout_seconds: float | None = None,
    ) -> LLMResult:
        with self._observability.observe(component="llm", operation="complete") as observed:
            result = await self._backend.complete(messages, response_schema, timeout_seconds)
            observed.set_attribute("usage.prompt_tokens", result.usage.prompt_tokens)
            observed.set_attribute("usage.completion_tokens", result.usage.completion_tokens)
            observed.set_attribute("usage.total_tokens", result.usage.total_tokens)
            observed.set_log_field("total_tokens", result.usage.total_tokens)
            self._observability.record_usage(result.usage)
            return result


class ObservedRetriever:
    def __init__(self, backend: Retriever, observability: Observability) -> None:
        self._backend = backend
        self._observability = observability

    async def aretrieve(
        self,
        query: str,
        auth_filter: AuthFilter,
        top_k: int,
    ) -> list[RetrievedChunk]:
        with self._observability.observe(component="retrieval", operation="retrieve") as observed:
            chunks = await self._backend.aretrieve(query, auth_filter, top_k)
            observed.set_attribute("result.count", len(chunks))
            observed.set_log_field("result_count", len(chunks))
            self._observability.metrics.record_result_items(
                component="retrieval", operation="retrieve", count=len(chunks)
            )
            return chunks


class ObservedReranker:
    def __init__(self, backend: Reranker, observability: Observability) -> None:
        self._backend = backend
        self._observability = observability

    async def arerank(
        self,
        query: str,
        chunks: Sequence[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        with self._observability.observe(component="rerank", operation="rerank") as observed:
            result = await self._backend.arerank(query, chunks, top_k)
            observed.set_attribute("result.count", len(result))
            observed.set_log_field("result_count", len(result))
            self._observability.metrics.record_result_items(
                component="rerank", operation="rerank", count=len(result)
            )
            return result


class ObservedPolicyEngine:
    def __init__(self, backend: PolicyEngine, observability: Observability) -> None:
        self._backend = backend
        self._observability = observability

    def authorize_read(
        self,
        principal: Principal,
        document: DocumentMeta,
        *,
        workspace_authorized: bool,
    ) -> bool:
        with self._observability.observe(
            component="policy", operation="authorize_read"
        ) as observed:
            allowed = self._backend.authorize_read(
                principal, document, workspace_authorized=workspace_authorized
            )
            result = "allowed" if allowed else "denied"
            observed.set_attribute("policy.result", result)
            observed.set_log_field("policy_result", result)
            return allowed

    def authorize_tool(self, principal: Principal, tool_name: str) -> bool:
        with self._observability.observe(
            component="policy", operation="authorize_tool"
        ) as observed:
            allowed = self._backend.authorize_tool(principal, tool_name)
            result = "allowed" if allowed else "denied"
            observed.set_attribute("policy.result", result)
            observed.set_attribute("tool.name", tool_name)
            observed.set_log_field("policy_result", result)
            observed.set_log_field("tool_name", tool_name)
            return allowed


def instrument_runtime_context(context: AgentRuntimeContext) -> AgentRuntimeContext:
    observability = context.observability
    llm = (
        context.llm
        if isinstance(context.llm, ObservedLLM)
        else ObservedLLM(context.llm, observability)
    )
    retriever = (
        context.retriever
        if isinstance(context.retriever, ObservedRetriever)
        else ObservedRetriever(context.retriever, observability)
    )
    policy = (
        context.policy_engine
        if isinstance(context.policy_engine, ObservedPolicyEngine)
        else ObservedPolicyEngine(context.policy_engine, observability)
    )
    return replace(context, llm=llm, retriever=retriever, policy_engine=policy)


def observed_async_node(name: str, node: AsyncNode) -> AsyncNode:
    @wraps(node)
    async def wrapped(
        state: AgentState,
        runtime: Runtime[AgentRuntimeContext],
    ) -> AgentState:
        observability = _runtime_observability(runtime)
        with observability.observe(
            component="agent",
            operation=name,
            trace_id=_state_trace_id(state),
            attributes=_state_attributes(state, name),
        ) as observed:
            result = await node(state, runtime)
            _record_node_result(observed, result)
            return result

    return wrapped


def observed_runtime_node(name: str, node: SyncRuntimeNode) -> SyncRuntimeNode:
    @wraps(node)
    def wrapped(
        state: AgentState,
        runtime: Runtime[AgentRuntimeContext],
    ) -> AgentState:
        observability = _runtime_observability(runtime)
        with observability.observe(
            component="agent",
            operation=name,
            trace_id=_state_trace_id(state),
            attributes=_state_attributes(state, name),
        ) as observed:
            result = node(state, runtime)
            _record_node_result(observed, result)
            return result

    return wrapped


def observed_state_node(name: str, node: SyncStateNode) -> SyncRuntimeNode:
    def wrapped(
        state: AgentState,
        runtime: Runtime[AgentRuntimeContext],
    ) -> AgentState:
        observability = _runtime_observability(runtime)
        with observability.observe(
            component="agent",
            operation=name,
            trace_id=_state_trace_id(state),
            attributes=_state_attributes(state, name),
        ) as observed:
            result = node(state)
            _record_node_result(observed, result)
            return result

    wrapped.__name__ = node.__name__
    return wrapped


def _runtime_observability(runtime: Runtime[AgentRuntimeContext]) -> Observability:
    if runtime.context is None:
        from app.observability.tracing import get_observability

        return get_observability()
    return runtime.context.observability


def _state_trace_id(state: AgentState) -> str:
    return state.get("trace_id") or current_trace_id() or "background"


def _state_attributes(state: AgentState, name: str) -> dict[str, str | int]:
    attributes: dict[str, str | int] = {
        "node": name,
        "retry.count": state.get("retry_count", 0),
    }
    route = state.get("route")
    if route:
        attributes["route"] = route
    return attributes


def _record_node_result(observed: object, result: AgentState) -> None:
    from app.observability.tracing import Observation

    if not isinstance(observed, Observation):
        return
    route = result.get("route")
    if route:
        observed.set_attribute("route", route)
        observed.set_log_field("route", route)
    retry_count = result.get("retry_count")
    if retry_count is not None:
        observed.set_attribute("retry.count", retry_count)
        observed.set_log_field("retry_count", retry_count)
