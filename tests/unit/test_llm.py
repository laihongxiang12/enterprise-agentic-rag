"""Deterministic tests for structured LLM output, retries, usage, and isolation."""

import asyncio
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from openai import APIConnectionError
from pydantic import ValidationError

from app.core.config import Settings
from app.core.dependencies import build_llm_provider
from app.llm.base import (
    LLMBudgetExceeded,
    LLMConfigurationError,
    LLMMessage,
    LLMOutputError,
    LLMTransientError,
)
from app.llm.budgets import TokenPrice
from app.llm.fake import FakeLLM
from app.llm.providers.deepseek import DeepSeekLLM
from app.llm.providers.openai_compatible import OpenAICompatibleLLM
from app.schemas.agent import EvidenceGrade, RouteDecision


def response(
    content: str | None,
    *,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


class StubCompletions:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = outcomes
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class ConcurrencyCompletions:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0

    async def create(self, **kwargs: Any) -> Any:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        return response("ok")


def stub_client(*outcomes: Any) -> tuple[SimpleNamespace, StubCompletions]:
    completions = StubCompletions(list(outcomes))
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


def provider(
    *outcomes: Any,
    max_attempts: int = 3,
    max_calls: int = 12,
    price: TokenPrice | None = None,
) -> tuple[OpenAICompatibleLLM, StubCompletions]:
    client, completions = stub_client(*outcomes)
    return (
        OpenAICompatibleLLM(
            api_key="test-key",
            base_url="https://llm.invalid/v1",
            model="test-model",
            max_attempts=max_attempts,
            max_calls=max_calls,
            retry_base_seconds=0,
            retry_jitter_seconds=0,
            price=price,
            client=client,
        ),
        completions,
    )


@pytest.mark.parametrize(
    ("schema", "payload"),
    [
        (
            RouteDecision,
            '{"route":"knowledge","confidence":0.9,"reason":"policy question"}',
        ),
        (
            EvidenceGrade,
            '{"relevant":true,"sufficient":false,"confidence":0.7,'
            '"missing_aspects":["limit"],"recommended_action":"rewrite",'
            '"reason":"limit missing"}',
        ),
    ],
)
async def test_fake_llm_validates_agent_structures(schema: Any, payload: str) -> None:
    llm = FakeLLM(responses={"question": payload})

    result = await llm.complete([LLMMessage(role="user", content="question")], schema)

    assert isinstance(result.parsed, schema)
    assert result.usage.estimated_cost is None


@pytest.mark.parametrize(
    "payload",
    [
        '{"route":"unsupported","confidence":0.9,"reason":"bad"}',
        '{"route":"knowledge","confidence":2,"reason":"bad"}',
        '{"route":"knowledge"',
    ],
)
async def test_fake_llm_rejects_invalid_route_schema(payload: str) -> None:
    llm = FakeLLM(responses={"question": payload})

    with pytest.raises(LLMOutputError):
        await llm.complete([LLMMessage(role="user", content="question")], RouteDecision)


async def test_structured_output_retries_and_aggregates_usage() -> None:
    llm, completions = provider(
        response('{"route":"knowledge"', prompt_tokens=4, completion_tokens=2),
        response(
            '{"route":"direct","confidence":0.8,"reason":"general"}',
            prompt_tokens=5,
            completion_tokens=3,
        ),
        price=TokenPrice(Decimal("1"), Decimal("2")),
    )

    result = await llm.complete(
        [LLMMessage(role="user", content="hello")],
        RouteDecision,
    )

    assert isinstance(result.parsed, RouteDecision)
    assert result.usage.model_dump() == {
        "prompt_tokens": 9,
        "completion_tokens": 5,
        "total_tokens": 14,
        "estimated_cost": 0.000019,
        "cost_is_estimate": True,
    }
    assert len(completions.calls) == 2
    assert completions.calls[0]["response_format"]["type"] == "json_schema"


async def test_unknown_price_stays_none() -> None:
    llm, _ = provider(response("ok"))

    result = await llm.complete([LLMMessage(role="user", content="hello")])

    assert result.usage.total_tokens == 15
    assert result.usage.estimated_cost is None
    assert result.usage.cost_is_estimate is False


async def test_empty_output_exhausts_exact_attempt_limit() -> None:
    llm, completions = provider(response(""), response(None), max_attempts=2)

    with pytest.raises(LLMOutputError, match="bounded retries"):
        await llm.complete([LLMMessage(role="user", content="hello")])

    assert len(completions.calls) == 2


class StatusFailure(RuntimeError):
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}")


@pytest.mark.parametrize(
    "failure",
    [TimeoutError(), StatusFailure(429), StatusFailure(500)],
    ids=["timeout", "rate-limit", "server-error"],
)
async def test_retryable_failures_are_bounded(failure: Exception) -> None:
    llm, completions = provider(failure, failure, max_attempts=2)

    with pytest.raises(LLMTransientError):
        await llm.complete([LLMMessage(role="user", content="hello")])

    assert len(completions.calls) == 2


async def test_network_failure_is_bounded() -> None:
    request = httpx.Request("POST", "https://llm.invalid/v1/chat/completions")
    failures = [APIConnectionError(request=request) for _ in range(2)]
    llm, completions = provider(*failures, max_attempts=2)

    with pytest.raises(LLMTransientError):
        await llm.complete([LLMMessage(role="user", content="hello")])

    assert len(completions.calls) == 2


async def test_non_retryable_client_error_is_not_retried() -> None:
    llm, completions = provider(StatusFailure(400), response("unused"))

    with pytest.raises(StatusFailure):
        await llm.complete([LLMMessage(role="user", content="hello")])

    assert len(completions.calls) == 1


async def test_external_call_budget_counts_each_attempt() -> None:
    llm, completions = provider(
        StatusFailure(429),
        response("would exceed budget"),
        max_calls=1,
    )

    with pytest.raises(LLMBudgetExceeded):
        await llm.complete([LLMMessage(role="user", content="hello")])

    assert len(completions.calls) == 1
    assert (await llm.budget.snapshot()).calls == 1


async def test_semaphore_caps_concurrent_external_calls() -> None:
    completions = ConcurrencyCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    llm = OpenAICompatibleLLM(
        api_key="test-key",
        base_url="https://llm.invalid/v1",
        model="test-model",
        max_concurrency=1,
        retry_base_seconds=0,
        retry_jitter_seconds=0,
        client=client,
    )
    messages = [LLMMessage(role="user", content="hello")]

    await asyncio.gather(*(llm.complete(messages) for _ in range(3)))

    assert completions.max_active == 1


async def test_deepseek_uses_documented_json_object_mode() -> None:
    client, completions = stub_client(
        response('{"route":"direct","confidence":1,"reason":"simple"}')
    )
    llm = DeepSeekLLM(
        api_key="test-key",
        base_url="https://api.deepseek.com",
        model="deepseek-flash",
        retry_base_seconds=0,
        retry_jitter_seconds=0,
        client=client,
    )

    await llm.complete([LLMMessage(role="user", content="hello")], RouteDecision)

    call = completions.calls[0]
    assert call["response_format"] == {"type": "json_object"}
    assert "json" in call["messages"][0]["content"].lower()


def test_api_key_is_required_before_client_construction() -> None:
    with pytest.raises(LLMConfigurationError, match="API key"):
        OpenAICompatibleLLM(
            api_key=" ",
            base_url="https://llm.invalid/v1",
            model="test-model",
        )


def test_fake_and_real_modes_are_separate() -> None:
    fake = build_llm_provider(Settings(app_mode="fake", llm_api_key=None))
    local = build_llm_provider(Settings(app_mode="local", llm_api_key=None))

    assert isinstance(fake, FakeLLM)
    assert isinstance(local, FakeLLM)
    with pytest.raises(ValidationError, match="LLM_API_KEY"):
        Settings(app_mode="real", llm_api_key=None)
