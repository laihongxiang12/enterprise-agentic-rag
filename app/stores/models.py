"""Persistence boundary models without sensitive document payloads."""

from typing import Annotated, Literal

from pydantic import Field

from app.schemas.base import ContractModel

NonEmptyString = Annotated[str, Field(min_length=1)]
AuditValue = str | int | bool | None


class IngestJob(ContractModel):
    tenant_id: NonEmptyString
    workspace_id: NonEmptyString
    job_id: NonEmptyString
    document_id: NonEmptyString
    status: Literal["pending", "running", "succeeded", "failed"] = "pending"
    error_code: str | None = None


class AuditEvent(ContractModel):
    tenant_id: NonEmptyString
    event_id: NonEmptyString
    actor_user_id: NonEmptyString
    action: NonEmptyString
    resource_type: NonEmptyString
    resource_id: NonEmptyString
    outcome: Literal["allowed", "denied", "success", "failure"]
    trace_id: NonEmptyString
    details: dict[str, AuditValue] = Field(default_factory=dict)
