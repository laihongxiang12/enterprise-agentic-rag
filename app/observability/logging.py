"""Structured JSON events with an intentionally narrow field vocabulary."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import TextIO

SafeLogValue = str | int | float | bool | None

_SAFE_FIELDS = frozenset(
    {
        "cache_hit",
        "duration_ms",
        "error_type",
        "finish_reason",
        "http_method",
        "http_route",
        "http_status_code",
        "node",
        "policy_result",
        "result_count",
        "retry_count",
        "route",
        "tool_name",
        "total_tokens",
    }
)


def build_json_logger(
    *,
    level: str = "INFO",
    stream: TextIO | None = None,
) -> logging.Logger:
    """Build an isolated logger that writes exactly one JSON object per event."""

    logger = logging.Logger("enterprise_agentic_rag", level=level)
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


class JsonLogEmitter:
    """Emit allowlisted metadata only; arbitrary application payloads are rejected."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or build_json_logger()

    def emit(
        self,
        *,
        event_type: str,
        trace_id: str,
        component: str,
        operation: str,
        outcome: str,
        fields: dict[str, SafeLogValue] | None = None,
    ) -> None:
        supplied = fields or {}
        unknown = set(supplied).difference(_SAFE_FIELDS)
        if unknown:
            raise ValueError(f"unsafe structured log fields: {sorted(unknown)}")
        payload: dict[str, SafeLogValue] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event_type": _label(event_type),
            "trace_id": _trace_id(trace_id),
            "component": _label(component),
            "operation": _label(operation),
            "outcome": _label(outcome),
        }
        payload.update(supplied)
        self._logger.info(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


def _label(value: str) -> str:
    normalized = value.strip()
    if (
        not normalized
        or len(normalized) > 80
        or not normalized.replace("_", "").replace(".", "").replace("-", "").isalnum()
    ):
        raise ValueError("log labels must be bounded identifiers")
    return normalized


def _trace_id(value: str) -> str:
    normalized = value.strip()
    if (
        not normalized
        or len(normalized) > 128
        or any(not character.isprintable() for character in normalized)
    ):
        raise ValueError("trace_id is invalid")
    return normalized
