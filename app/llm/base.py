"""Narrow asynchronous LLM boundary."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from app.schemas.api import Usage
from app.schemas.base import ContractModel


class LLMMessage(ContractModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = Field(min_length=1)


@dataclass(frozen=True)
class LLMResult:
    content: str
    usage: Usage
    parsed: BaseModel | None = None


@runtime_checkable
class LLMProvider(Protocol):
    async def complete(
        self,
        messages: Sequence[LLMMessage],
        response_schema: type[BaseModel] | None = None,
        timeout_seconds: float | None = None,
    ) -> LLMResult: ...
