"""HTTP query request and response contracts."""

from typing import Annotated, Literal

from pydantic import Field

from app.schemas.base import ContractModel

NonEmptyString = Annotated[str, Field(min_length=1)]


class Citation(ContractModel):
    document_id: NonEmptyString
    document_version: int = Field(ge=1)
    chunk_id: NonEmptyString
    title: NonEmptyString
    source_uri: NonEmptyString
    snippet: NonEmptyString
    page: int | None = None


class Usage(ContractModel):
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    estimated_cost: float | None = Field(default=None, ge=0)
    cost_is_estimate: bool = True


class QueryRequest(ContractModel):
    query: str = Field(min_length=1, max_length=8000)
    session_id: str | None = None
    mode: Literal["auto", "knowledge", "direct"] = "auto"
    workspace_id: str | None = None
    max_retries: int = Field(default=2, ge=0, le=3)
    debug: bool = False


class QueryResponse(ContractModel):
    trace_id: NonEmptyString
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    usage: Usage
    latency_ms: int = Field(ge=0)
    retries: int = Field(ge=0)
    finish_reason: Literal[
        "completed",
        "insufficient_evidence",
        "policy_blocked",
        "error",
    ]
