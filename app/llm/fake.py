"""Deterministic, network-free LLM adapter."""

from collections.abc import Mapping, Sequence

from pydantic import BaseModel

from app.llm.base import LLMMessage, LLMResult
from app.schemas.api import Usage


class FakeLLM:
    """Return configured text by final message content without external I/O."""

    def __init__(
        self,
        responses: Mapping[str, str] | None = None,
        *,
        default_response: str = "fake-response",
    ) -> None:
        self._responses = dict(responses or {})
        self._default_response = default_response
        self.calls: list[tuple[LLMMessage, ...]] = []

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        response_schema: type[BaseModel] | None = None,
        timeout_seconds: float | None = None,
    ) -> LLMResult:
        if not messages:
            raise ValueError("messages must not be empty")
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        copied_messages = tuple(message.model_copy(deep=True) for message in messages)
        self.calls.append(copied_messages)
        content = self._responses.get(messages[-1].content, self._default_response)
        parsed = (
            response_schema.model_validate_json(content)
            if response_schema is not None
            else None
        )
        return LLMResult(
            content=content,
            parsed=parsed,
            usage=Usage(cost_is_estimate=True),
        )
