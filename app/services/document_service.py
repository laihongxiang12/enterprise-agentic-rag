"""Document upload workflow with bounded validation and an asynchronous queue seam."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal, Protocol
from uuid import uuid4

from app.core.errors import AppError
from app.schemas.documents import (
    DocumentDeleteResponse,
    DocumentStatusResponse,
    DocumentUploadResponse,
    Principal,
)

Visibility = Literal["restricted", "tenant"]
_MIME_BY_SUFFIX = {
    ".html": frozenset({"text/html", "application/xhtml+xml"}),
    ".htm": frozenset({"text/html", "application/xhtml+xml"}),
    ".md": frozenset({"text/markdown", "text/x-markdown", "text/plain"}),
    ".markdown": frozenset({"text/markdown", "text/x-markdown", "text/plain"}),
    ".pdf": frozenset({"application/pdf"}),
}


@dataclass(frozen=True)
class DocumentUpload:
    workspace_id: str
    filename: str
    content_type: str
    content: bytes
    visibility: Visibility
    acl_groups: tuple[str, ...]
    idempotency_key: str | None


@dataclass(frozen=True)
class QueuedDocument:
    tenant_id: str
    workspace_id: str
    document_id: str
    job_id: str
    filename: str
    content_type: str
    content: bytes
    visibility: Visibility
    acl_groups: tuple[str, ...]


class IngestQueue(Protocol):
    async def enqueue(self, document: QueuedDocument) -> None: ...


class MemoryIngestQueue:
    """Deterministic fake queue; payloads are retained only for test inspection."""

    def __init__(self) -> None:
        self.items: list[QueuedDocument] = []

    async def enqueue(self, document: QueuedDocument) -> None:
        self.items.append(document)


@dataclass
class _Record:
    response: DocumentStatusResponse
    fingerprint: str
    visibility: Visibility
    acl_groups: tuple[str, ...]


class InMemoryDocumentService:
    """Local/fake service implementing the same boundary as a durable worker setup."""

    def __init__(
        self,
        queue: IngestQueue,
        *,
        workspace_memberships: set[tuple[str, str, str]],
        max_bytes: int,
    ) -> None:
        self._queue = queue
        self._memberships = workspace_memberships
        self._max_bytes = max_bytes
        self._records: dict[tuple[str, str, str], _Record] = {}
        self._idempotency: dict[tuple[str, str, str], tuple[str, str]] = {}

    async def upload(self, principal: Principal, upload: DocumentUpload) -> DocumentUploadResponse:
        self._require_scope(principal, "documents:write")
        self._require_workspace(principal, upload.workspace_id)
        content_type = upload.content_type.partition(";")[0].strip().lower()
        suffix = _suffix(upload.filename)
        if suffix not in _MIME_BY_SUFFIX or content_type not in _MIME_BY_SUFFIX[suffix]:
            raise AppError(
                code="unsupported_document_type",
                message="File extension and content type must describe PDF, Markdown, or HTML",
                status_code=422,
            )
        if not upload.content:
            raise AppError(code="empty_file", message="Uploaded file is empty", status_code=422)
        if len(upload.content) > self._max_bytes:
            raise AppError(
                code="file_too_large",
                message="Uploaded file exceeds the configured size limit",
                status_code=413,
            )
        if suffix == ".pdf" and not upload.content.startswith(b"%PDF-"):
            raise AppError(
                code="mime_mismatch",
                message="PDF signature does not match the file extension",
                status_code=422,
            )
        acl_groups = tuple(sorted({group.strip() for group in upload.acl_groups if group.strip()}))
        if upload.visibility == "restricted" and not acl_groups:
            raise AppError(
                code="acl_required",
                message="Restricted documents require at least one ACL group",
                status_code=422,
            )
        if "documents:admin" not in principal.scopes and not set(acl_groups).issubset(
            principal.group_ids
        ):
            raise AppError(
                code="forbidden_acl",
                message="Caller cannot assign an ACL group they do not belong to",
                status_code=403,
            )

        fingerprint = hashlib.sha256(
            b"\x1f".join(
                (
                    upload.filename.encode(),
                    content_type.encode(),
                    upload.visibility.encode(),
                    ",".join(acl_groups).encode(),
                    upload.content,
                )
            )
        ).hexdigest()
        if upload.idempotency_key:
            key = (principal.tenant_id, upload.workspace_id, upload.idempotency_key)
            existing = self._idempotency.get(key)
            if existing is not None:
                existing_document_id, existing_fingerprint = existing
                if existing_fingerprint != fingerprint:
                    raise AppError(
                        code="idempotency_conflict",
                        message="Idempotency key was already used for a different upload",
                        status_code=409,
                    )
                return self._upload_response(
                    self._records[(principal.tenant_id, upload.workspace_id, existing_document_id)]
                )

        document_id = str(uuid4())
        job_id = str(uuid4())
        status = DocumentStatusResponse(
            document_id=document_id,
            job_id=job_id,
            status="pending",
            workspace_id=upload.workspace_id,
            filename=upload.filename,
            content_type=content_type,
        )
        record = _Record(status, fingerprint, upload.visibility, acl_groups)
        self._records[(principal.tenant_id, upload.workspace_id, document_id)] = record
        if upload.idempotency_key:
            self._idempotency[
                (principal.tenant_id, upload.workspace_id, upload.idempotency_key)
            ] = (
                document_id,
                fingerprint,
            )
        try:
            await self._queue.enqueue(
                QueuedDocument(
                    tenant_id=principal.tenant_id,
                    workspace_id=upload.workspace_id,
                    document_id=document_id,
                    job_id=job_id,
                    filename=upload.filename,
                    content_type=content_type,
                    content=upload.content,
                    visibility=upload.visibility,
                    acl_groups=acl_groups,
                )
            )
        except Exception:
            self._records.pop((principal.tenant_id, upload.workspace_id, document_id), None)
            if upload.idempotency_key:
                self._idempotency.pop(
                    (principal.tenant_id, upload.workspace_id, upload.idempotency_key), None
                )
            raise AppError(
                code="ingest_queue_unavailable",
                message="Document ingestion queue is unavailable",
                status_code=503,
            ) from None
        return self._upload_response(record)

    async def status(
        self, principal: Principal, workspace_id: str, document_id: str
    ) -> DocumentStatusResponse:
        self._require_scope(principal, "documents:read")
        self._require_workspace(principal, workspace_id)
        record = self._records.get((principal.tenant_id, workspace_id, document_id))
        if record is None or not self._can_access(principal, record):
            raise AppError(
                code="document_not_found", message="Document was not found", status_code=404
            )
        return record.response.model_copy(deep=True)

    async def delete(
        self, principal: Principal, workspace_id: str, document_id: str
    ) -> DocumentDeleteResponse:
        self._require_scope(principal, "documents:write")
        self._require_workspace(principal, workspace_id)
        key = (principal.tenant_id, workspace_id, document_id)
        record = self._records.get(key)
        if record is None or not self._can_access(principal, record):
            raise AppError(
                code="document_not_found", message="Document was not found", status_code=404
            )
        self._records.pop(key)
        for idempotency_key, value in list(self._idempotency.items()):
            if idempotency_key[:2] == key[:2] and value[0] == document_id:
                self._idempotency.pop(idempotency_key)
        return DocumentDeleteResponse(document_id=document_id)

    @staticmethod
    def _can_access(principal: Principal, record: _Record) -> bool:
        return bool(
            "documents:admin" in principal.scopes
            or record.visibility == "tenant"
            or set(record.acl_groups).intersection(principal.group_ids)
        )

    def _require_workspace(self, principal: Principal, workspace_id: str) -> None:
        if (principal.tenant_id, workspace_id, principal.user_id) not in self._memberships:
            raise AppError(
                code="workspace_forbidden",
                message="Caller is not a member of this workspace",
                status_code=403,
            )

    @staticmethod
    def _require_scope(principal: Principal, scope: str) -> None:
        if scope not in principal.scopes:
            raise AppError(code="forbidden", message="Required scope is missing", status_code=403)

    @staticmethod
    def _upload_response(record: _Record) -> DocumentUploadResponse:
        return DocumentUploadResponse(
            document_id=record.response.document_id,
            job_id=record.response.job_id,
            status=record.response.status,
        )


def _suffix(filename: str) -> str:
    normalized = filename.strip().lower()
    dot = normalized.rfind(".")
    return "" if dot < 0 else normalized[dot:]
