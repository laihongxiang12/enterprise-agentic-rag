"""OpenTelemetry spans and per-trace usage aggregation without sensitive payloads."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache
from threading import Lock

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.trace import Span, Status, StatusCode, Tracer

from app.observability.logging import JsonLogEmitter, SafeLogValue
from app.observability.metrics import MetricsRegistry
from app.schemas.api import Usage

SafeAttribute = str | int | float | bool

_SAFE_ATTRIBUTES = frozenset(
    {
        "app.trace_id",
        "cache.hit",
        "component",
        "error.type",
        "finish_reason",
        "http.request.method",
        "http.response.status_code",
        "http.route",
        "node",
        "operation",
        "policy.result",
        "result.count",
        "retry.count",
        "route",
        "tool.name",
        "usage.completion_tokens",
        "usage.prompt_tokens",
        "usage.total_tokens",
    }
)
_TRACE_ID: ContextVar[str | None] = ContextVar("enterprise_rag_trace_id", default=None)


@dataclass(slots=True)
class _UsageAccumulator:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float | None = None
    cost_is_estimate: bool = False


_USAGE: ContextVar[_UsageAccumulator | None] = ContextVar("enterprise_rag_usage", default=None)


class BoundedSpanExporter(SpanExporter):
    """A bounded in-process exporter for local debugging and deterministic tests."""

    def __init__(self, max_spans: int = 1_000) -> None:
        if max_spans < 1:
            raise ValueError("max_spans must be positive")
        self._spans: deque[ReadableSpan] = deque(maxlen=max_spans)
        self._lock = Lock()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        with self._lock:
            self._spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        del timeout_millis
        return True

    def finished_spans(self) -> tuple[ReadableSpan, ...]:
        with self._lock:
            return tuple(self._spans)


class Observation:
    """Mutable safe metadata handle for one active span."""

    def __init__(self, span: Span) -> None:
        self._span = span
        self.log_fields: dict[str, SafeLogValue] = {}

    def set_attribute(self, key: str, value: SafeAttribute) -> None:
        if key not in _SAFE_ATTRIBUTES:
            raise ValueError(f"unsafe span attribute: {key}")
        if isinstance(value, str) and (
            len(value) > 128 or any(not character.isprintable() for character in value)
        ):
            raise ValueError("span string attributes must be bounded and printable")
        self._span.set_attribute(key, value)

    def set_log_field(self, key: str, value: SafeLogValue) -> None:
        self.log_fields[key] = value


class Observability:
    """One application-local OTel provider, metrics registry, and JSON emitter."""

    def __init__(
        self,
        *,
        metrics: MetricsRegistry | None = None,
        logs: JsonLogEmitter | None = None,
        max_finished_spans: int = 1_000,
    ) -> None:
        self.metrics = metrics or MetricsRegistry()
        self.logs = logs or JsonLogEmitter()
        self._exporter = BoundedSpanExporter(max_finished_spans)
        provider = TracerProvider(
            resource=Resource.create({"service.name": "enterprise-agentic-rag"})
        )
        provider.add_span_processor(SimpleSpanProcessor(self._exporter))
        self._provider = provider
        self._tracer: Tracer = trace.get_tracer("enterprise-agentic-rag", tracer_provider=provider)

    @contextmanager
    def trace_scope(self, trace_id: str) -> Iterator[None]:
        normalized = _validated_trace_id(trace_id)
        trace_token = _TRACE_ID.set(normalized)
        usage_token = _USAGE.set(_UsageAccumulator())
        try:
            yield
        finally:
            _USAGE.reset(usage_token)
            _TRACE_ID.reset(trace_token)

    @contextmanager
    def observe(
        self,
        *,
        component: str,
        operation: str,
        trace_id: str | None = None,
        attributes: Mapping[str, SafeAttribute] | None = None,
    ) -> Iterator[Observation]:
        resolved_trace = _validated_trace_id(trace_id or current_trace_id() or "background")
        started = time.perf_counter()
        outcome = "success"
        error_type: str | None = None
        with self._tracer.start_as_current_span(f"{component}.{operation}") as span:
            observation = Observation(span)
            observation.set_attribute("app.trace_id", resolved_trace)
            observation.set_attribute("component", component)
            observation.set_attribute("operation", operation)
            for key, value in (attributes or {}).items():
                observation.set_attribute(key, value)
            try:
                yield observation
            except Exception as exc:
                outcome = "error"
                error_type = type(exc).__name__
                observation.set_attribute("error.type", error_type)
                span.set_status(Status(StatusCode.ERROR, error_type))
                raise
            finally:
                duration_seconds = max(0.0, time.perf_counter() - started)
                self.metrics.record_operation(
                    component=component,
                    operation=operation,
                    outcome=outcome,
                    duration_seconds=duration_seconds,
                )
                fields = dict(observation.log_fields)
                fields["duration_ms"] = round(duration_seconds * 1_000, 3)
                if error_type is not None:
                    fields["error_type"] = error_type
                self.logs.emit(
                    event_type="operation_completed",
                    trace_id=resolved_trace,
                    component=component,
                    operation=operation,
                    outcome=outcome,
                    fields=fields,
                )

    def record_usage(self, usage: Usage) -> None:
        if usage.total_tokens < usage.prompt_tokens + usage.completion_tokens:
            raise ValueError("provider usage totals are inconsistent")
        self.metrics.record_tokens(
            prompt=usage.prompt_tokens,
            completion=usage.completion_tokens,
            total=usage.total_tokens,
        )
        current = _USAGE.get()
        if current is None:
            return
        if current.calls == 0:
            cost = usage.estimated_cost
        elif current.estimated_cost is None or usage.estimated_cost is None:
            cost = None
        else:
            cost = current.estimated_cost + usage.estimated_cost
        # LangGraph may execute nodes in child asyncio contexts. Context variables are
        # copied into those tasks, so mutate the shared per-request accumulator rather
        # than replacing it; the parent request can then read the complete usage.
        current.calls += 1
        current.prompt_tokens += usage.prompt_tokens
        current.completion_tokens += usage.completion_tokens
        current.total_tokens += usage.total_tokens
        current.estimated_cost = cost
        current.cost_is_estimate = current.cost_is_estimate or usage.cost_is_estimate

    def current_usage(self) -> Usage:
        current = _USAGE.get() or _UsageAccumulator()
        return Usage(
            prompt_tokens=current.prompt_tokens,
            completion_tokens=current.completion_tokens,
            total_tokens=current.total_tokens,
            estimated_cost=current.estimated_cost,
            cost_is_estimate=current.cost_is_estimate,
        )

    def finished_spans(self) -> tuple[ReadableSpan, ...]:
        self._provider.force_flush()
        return self._exporter.finished_spans()


def current_trace_id() -> str | None:
    return _TRACE_ID.get()


@lru_cache(maxsize=1)
def get_observability() -> Observability:
    return Observability()


def _validated_trace_id(value: str) -> str:
    normalized = value.strip()
    if (
        not normalized
        or len(normalized) > 128
        or any(not character.isprintable() for character in normalized)
    ):
        raise ValueError("trace_id is invalid")
    return normalized
