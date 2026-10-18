"""Deterministic, network-free LLM adapter."""

from collections.abc import Mapping, Sequence

from pydantic import BaseModel

from app.llm.base import LLMMessage, LLMResult, parse_structured_output
from app.schemas.api import Usage


class FakeLLM:
    """Return scripted text or failures by final message content without I/O."""

    def __init__(
        self,
        responses: Mapping[str, str] | None = None,
        *,
        scripts: Mapping[str, Sequence[str | Exception]] | None = None,
        default_response: str = "fake-response",
        usage: Usage | None = None,
    ) -> None:
        self._responses = dict(responses or {})
        self._scripts = {key: list(values) for key, values in (scripts or {}).items()}
        self._default_response = default_response
        self._usage = usage or Usage(cost_is_estimate=False)
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
        key = messages[-1].content
        scripted = self._scripts.get(key)
        if scripted:
            outcome: str | Exception = scripted.pop(0)
        else:
            outcome = self._responses.get(key, self._default_response)
        if isinstance(outcome, Exception):
            raise outcome

        parsed = (
            parse_structured_output(outcome, response_schema)
            if response_schema is not None
            else None
        )
        return LLMResult(content=outcome, parsed=parsed, usage=self._usage.model_copy())
