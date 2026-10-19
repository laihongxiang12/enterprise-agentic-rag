"""Prometheus metrics with a fixed, low-cardinality label set."""

from __future__ import annotations

import re

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram

PROMETHEUS_CONTENT_TYPE = CONTENT_TYPE_LATEST
_LABEL = re.compile(r"[a-z][a-z0-9_.-]{0,63}")


class MetricsRegistry:
    """Own an isolated registry so tests and app instances never share counters."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry(auto_describe=True)
        self.operations = Counter(
            "enterprise_rag_operations_total",
            "Completed bounded operations",
            ("component", "operation", "outcome"),
            registry=self.registry,
        )
        self.duration = Histogram(
            "enterprise_rag_operation_duration_seconds",
            "Operation latency in seconds",
            ("component", "operation"),
            registry=self.registry,
            buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 30),
        )
        self.retries = Counter(
            "enterprise_rag_retries_total",
            "Agent retry count",
            ("component",),
            registry=self.registry,
        )
        self.tokens = Counter(
            "enterprise_rag_llm_tokens_total",
            "LLM tokens reported by providers",
            ("kind",),
            registry=self.registry,
        )
        self.result_items = Counter(
            "enterprise_rag_result_items_total",
            "Items returned by bounded data operations",
            ("component", "operation"),
            registry=self.registry,
        )

    def record_operation(
        self,
        *,
        component: str,
        operation: str,
        outcome: str,
        duration_seconds: float,
    ) -> None:
        labels = (_label(component), _label(operation), _label(outcome))
        self.operations.labels(*labels).inc()
        self.duration.labels(*labels[:2]).observe(max(0.0, duration_seconds))

    def record_retries(self, component: str, count: int) -> None:
        if count < 0:
            raise ValueError("retry count cannot be negative")
        if count:
            self.retries.labels(_label(component)).inc(count)

    def record_tokens(self, *, prompt: int, completion: int, total: int) -> None:
        if min(prompt, completion, total) < 0 or total < prompt + completion:
            raise ValueError("token counts are inconsistent")
        self.tokens.labels("prompt").inc(prompt)
        self.tokens.labels("completion").inc(completion)
        self.tokens.labels("total").inc(total)

    def record_result_items(self, *, component: str, operation: str, count: int) -> None:
        if count < 0:
            raise ValueError("result count cannot be negative")
        if count:
            self.result_items.labels(_label(component), _label(operation)).inc(count)

    def render(self) -> bytes:
        from prometheus_client import generate_latest

        return generate_latest(self.registry)


def _label(value: str) -> str:
    if _LABEL.fullmatch(value) is None:
        raise ValueError("metric labels must be bounded server identifiers")
    return value
