"""Narrow asynchronous LLM boundary and stable failure vocabulary."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field, ValidationError

from app.schemas.api import Usage
from app.schemas.base import ContractModel


class LLMError(RuntimeError):
    """Base class for failures at the application LLM boundary."""


class LLMConfigurationError(LLMError):
    """The selected provider cannot be constructed from configuration."""


class LLMTransientError(LLMError):
    """A retryable provider failure exhausted the bounded retry policy."""


class LLMOutputError(LLMError):
    """The provider returned empty or schema-invalid output."""


class LLMBudgetExceeded(LLMError):
    """The configured external-call or token budget has been exhausted."""


class LLMMessage(ContractModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = Field(min_length=1)


@dataclass(frozen=True)
class LLMResult:
    content: str
    usage: Usage
    parsed: BaseModel | None = None


def parse_structured_output(content: str, schema: type[BaseModel]) -> BaseModel:
    """Validate model JSON without ever accepting a permissive partial object."""
    if not content.strip():
        raise LLMOutputError("LLM returned empty content")
    try:
        return schema.model_validate_json(content)
    except ValidationError as exc:
        raise LLMOutputError("LLM output failed schema validation") from exc


@runtime_checkable
class LLMProvider(Protocol):
    async def complete(
        self,
        messages: Sequence[LLMMessage],
        response_schema: type[BaseModel] | None = None,
        timeout_seconds: float | None = None,
    ) -> LLMResult: ...
