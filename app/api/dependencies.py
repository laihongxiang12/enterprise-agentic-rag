"""Trusted HTTP dependencies and replaceable API service contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from fastapi import Request

from app.core.errors import AppError
from app.schemas.api import QueryRequest, QueryResponse
from app.schemas.documents import (
    DocumentDeleteResponse,
    DocumentStatusResponse,
    DocumentUploadResponse,
    Principal,
)
from app.services.document_service import DocumentUpload


class Authenticator(Protocol):
    def verify_bearer(self, authorization: str | None) -> Principal: ...


class DocumentOperations(Protocol):
    async def upload(
        self, principal: Principal, upload: DocumentUpload
    ) -> DocumentUploadResponse: ...

    async def status(
        self, principal: Principal, workspace_id: str, document_id: str
    ) -> DocumentStatusResponse: ...

    async def delete(
        self, principal: Principal, workspace_id: str, document_id: str
    ) -> DocumentDeleteResponse: ...


class QueryOperations(Protocol):
    async def query(
        self,
        request: QueryRequest,
        principal: Principal,
        *,
        trace_id: str,
    ) -> QueryResponse: ...


class RequestLimiter(Protocol):
    async def check(self, principal: Principal, operation: str) -> None: ...


class NoopRequestLimiter:
    """Stage-15 seam; the durable Redis implementation belongs to stage 16."""

    async def check(self, principal: Principal, operation: str) -> None:
        del principal, operation


@dataclass(frozen=True)
class ApiServices:
    authenticator: Authenticator
    documents: DocumentOperations
    queries: QueryOperations
    limiter: RequestLimiter
    max_upload_bytes: int = 10 * 1024 * 1024


def get_api_services(request: Request) -> ApiServices:
    services = getattr(request.app.state, "api_services", None)
    if not isinstance(services, ApiServices):
        raise AppError(
            code="service_unavailable",
            message="API service dependencies are not configured",
            status_code=503,
        )
    return services


def get_principal(request: Request) -> Principal:
    principal = getattr(request.state, "principal", None)
    if not isinstance(principal, Principal):
        raise AppError(code="unauthorized", message="Authentication is required", status_code=401)
    return principal
