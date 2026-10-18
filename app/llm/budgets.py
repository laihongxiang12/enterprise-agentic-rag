"""Concurrency-safe request budgets and independently configurable pricing."""

import asyncio
from dataclasses import dataclass
from decimal import Decimal

from app.llm.base import LLMBudgetExceeded
from app.schemas.api import Usage

_MILLION = Decimal(1_000_000)


@dataclass(frozen=True)
class TokenPrice:
    """Per-million-token prices; both values are required to estimate cost."""

    input_per_million: Decimal | None = None
    output_per_million: Decimal | None = None

    def estimate(self, *, prompt_tokens: int, completion_tokens: int) -> float | None:
        if self.input_per_million is None or self.output_per_million is None:
            return None
        cost = (
            Decimal(prompt_tokens) * self.input_per_million
            + Decimal(completion_tokens) * self.output_per_million
        ) / _MILLION
        return float(cost)


def token_usage(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int | None,
    price: TokenPrice,
) -> Usage:
    """Normalize provider usage while keeping unknown cost explicitly unknown."""
    resolved_total = prompt_tokens + completion_tokens if total_tokens is None else total_tokens
    estimated_cost = price.estimate(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=resolved_total,
        estimated_cost=estimated_cost,
        cost_is_estimate=estimated_cost is not None,
    )


def add_usage(left: Usage, right: Usage) -> Usage:
    """Aggregate all attempts, including attempts with invalid model output."""
    if left.estimated_cost is None or right.estimated_cost is None:
        estimated_cost = None
    else:
        estimated_cost = left.estimated_cost + right.estimated_cost
    return Usage(
        prompt_tokens=left.prompt_tokens + right.prompt_tokens,
        completion_tokens=left.completion_tokens + right.completion_tokens,
        total_tokens=left.total_tokens + right.total_tokens,
        estimated_cost=estimated_cost,
        cost_is_estimate=estimated_cost is not None,
    )


@dataclass(frozen=True)
class BudgetSnapshot:
    calls: int
    total_tokens: int


class RequestBudget:
    """Atomically cap external attempts and block calls after a token ceiling."""

    def __init__(self, *, max_calls: int, max_total_tokens: int | None = None) -> None:
        if max_calls < 1:
            raise ValueError("max_calls must be positive")
        if max_total_tokens is not None and max_total_tokens < 1:
            raise ValueError("max_total_tokens must be positive")
        self._max_calls = max_calls
        self._max_total_tokens = max_total_tokens
        self._calls = 0
        self._total_tokens = 0
        self._lock = asyncio.Lock()

    async def reserve_call(self) -> None:
        async with self._lock:
            if self._calls >= self._max_calls:
                raise LLMBudgetExceeded("LLM external-call budget exhausted")
            if self._max_total_tokens is not None and self._total_tokens >= self._max_total_tokens:
                raise LLMBudgetExceeded("LLM token budget exhausted")
            self._calls += 1

    async def record(self, usage: Usage) -> None:
        async with self._lock:
            self._total_tokens += usage.total_tokens

    async def snapshot(self) -> BudgetSnapshot:
        async with self._lock:
            return BudgetSnapshot(calls=self._calls, total_tokens=self._total_tokens)
