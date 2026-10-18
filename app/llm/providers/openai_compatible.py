"""Bounded async adapter for OpenAI-compatible Chat Completions APIs."""

import asyncio
import random
import re
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Literal

from openai import APIConnectionError, APITimeoutError, AsyncOpenAI
from pydantic import BaseModel

from app.llm.base import (
    LLMConfigurationError,
    LLMMessage,
    LLMOutputError,
    LLMResult,
    LLMTransientError,
    parse_structured_output,
)
from app.llm.budgets import RequestBudget, TokenPrice, add_usage, token_usage
from app.schemas.api import Usage

StructuredOutputMode = Literal["json_schema", "json_object"]
Sleep = Callable[[float], Awaitable[None]]


class OpenAICompatibleLLM:
    """Use one bounded retry loop around an SDK client with retries disabled."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        price: TokenPrice | None = None,
        structured_output_mode: StructuredOutputMode = "json_schema",
        timeout_seconds: float = 30.0,
        max_attempts: int = 3,
        max_concurrency: int = 4,
        max_calls: int = 12,
        max_total_tokens: int | None = None,
        max_output_tokens: int = 1_024,
        retry_base_seconds: float = 0.25,
        retry_max_seconds: float = 2.0,
        retry_jitter_seconds: float = 0.1,
        client: Any | None = None,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        if not api_key.strip():
            raise LLMConfigurationError("LLM API key is required")
        if not base_url.strip() or not model.strip():
            raise LLMConfigurationError("LLM base URL and model are required")
        if timeout_seconds <= 0 or max_attempts < 1 or max_concurrency < 1:
            raise ValueError("timeout, attempts, and concurrency must be positive")
        if max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        self._model = model
        self._price = price or TokenPrice()
        self._structured_output_mode = structured_output_mode
        self._timeout_seconds = timeout_seconds
        self._max_attempts = max_attempts
        self._max_output_tokens = max_output_tokens
        self._retry_base_seconds = max(0.0, retry_base_seconds)
        self._retry_max_seconds = max(0.0, retry_max_seconds)
        self._retry_jitter_seconds = max(0.0, retry_jitter_seconds)
        self._sleep = sleep
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._budget = RequestBudget(max_calls=max_calls, max_total_tokens=max_total_tokens)
        self._client: Any = client or AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=0,
            timeout=timeout_seconds,
        )

    @property
    def budget(self) -> RequestBudget:
        return self._budget

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        response_schema: type[BaseModel] | None = None,
        timeout_seconds: float | None = None,
    ) -> LLMResult:
        if not messages:
            raise ValueError("messages must not be empty")
        timeout = self._timeout_seconds if timeout_seconds is None else timeout_seconds
        if timeout <= 0:
            raise ValueError("timeout_seconds must be positive")

        request_messages, response_format = self._request_contract(messages, response_schema)
        accumulated: Usage | None = None
        last_output_error: LLMOutputError | None = None

        for attempt in range(self._max_attempts):
            await self._budget.reserve_call()
            try:
                async with self._semaphore:
                    request_kwargs: dict[str, Any] = {
                        "model": self._model,
                        "messages": request_messages,
                        "max_tokens": self._max_output_tokens,
                        "timeout": timeout,
                    }
                    if response_format is not None:
                        request_kwargs["response_format"] = response_format
                    response = await asyncio.wait_for(
                        self._client.chat.completions.create(**request_kwargs),
                        timeout=timeout,
                    )
            except Exception as exc:
                if not self._is_retryable(exc):
                    raise
                if attempt + 1 >= self._max_attempts:
                    raise LLMTransientError("LLM request failed after bounded retries") from exc
                await self._backoff(attempt)
                continue

            current_usage = self._extract_usage(response)
            await self._budget.record(current_usage)
            accumulated = (
                current_usage if accumulated is None else add_usage(accumulated, current_usage)
            )
            try:
                content = self._extract_content(response)
                parsed = (
                    parse_structured_output(content, response_schema)
                    if response_schema is not None
                    else None
                )
            except LLMOutputError as exc:
                last_output_error = exc
                if attempt + 1 >= self._max_attempts:
                    break
                await self._backoff(attempt)
                continue
            return LLMResult(content=content, parsed=parsed, usage=accumulated)

        raise LLMOutputError(
            "LLM output remained invalid after bounded retries"
        ) from last_output_error

    def _request_contract(
        self,
        messages: Sequence[LLMMessage],
        response_schema: type[BaseModel] | None,
    ) -> tuple[list[dict[str, str]], dict[str, Any] | None]:
        request_messages = [message.model_dump() for message in messages]
        if response_schema is None:
            return request_messages, None
        schema = response_schema.model_json_schema()
        if self._structured_output_mode == "json_schema":
            name = re.sub(r"[^a-zA-Z0-9_-]", "_", response_schema.__name__)[:64]
            return request_messages, {
                "type": "json_schema",
                "json_schema": {"name": name, "strict": True, "schema": schema},
            }
        request_messages.insert(
            0,
            {
                "role": "system",
                "content": (
                    f"Return json only. The response must match this JSON Schema exactly: {schema}"
                ),
            },
        )
        return request_messages, {"type": "json_object"}

    def _extract_content(self, response: Any) -> str:
        choices = getattr(response, "choices", None)
        if not choices:
            raise LLMOutputError("LLM response contained no choices")
        content = getattr(getattr(choices[0], "message", None), "content", None)
        if not isinstance(content, str) or not content.strip():
            raise LLMOutputError("LLM returned empty content")
        return content

    def _extract_usage(self, response: Any) -> Usage:
        usage = getattr(response, "usage", None)
        if usage is None:
            return Usage(cost_is_estimate=False)
        return token_usage(
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            total_tokens=getattr(usage, "total_tokens", None),
            price=self._price,
        )

    def _is_retryable(self, exc: Exception) -> bool:
        if isinstance(
            exc,
            (TimeoutError, APITimeoutError, APIConnectionError),
        ):
            return True
        status_code = getattr(exc, "status_code", None)
        return status_code == 429 or (isinstance(status_code, int) and status_code >= 500)

    async def _backoff(self, attempt: int) -> None:
        delay = min(self._retry_max_seconds, self._retry_base_seconds * (2**attempt))
        if self._retry_jitter_seconds:
            delay += random.uniform(0.0, self._retry_jitter_seconds)
        if delay:
            await self._sleep(delay)
