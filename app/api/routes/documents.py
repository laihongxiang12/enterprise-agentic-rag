"""Authenticated document upload and lifecycle endpoints."""

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, File, Form, Header, UploadFile, status

from app.api.dependencies import ApiServices, get_api_services, get_principal
from app.schemas.documents import (
    DocumentDeleteResponse,
    DocumentStatusResponse,
    DocumentUploadResponse,
    Principal,
)
from app.services.document_service import DocumentUpload

router = APIRouter(prefix="/v1/documents", tags=["documents"])


@router.post("", response_model=DocumentUploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    file: Annotated[UploadFile, File()],
    workspace_id: Annotated[str, Form(min_length=1)],
    principal: Annotated[Principal, Depends(get_principal)],
    services: Annotated[ApiServices, Depends(get_api_services)],
    visibility: Annotated[Literal["restricted", "tenant"], Form()] = "restricted",
    acl_groups: Annotated[list[str] | None, Form()] = None,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> DocumentUploadResponse:
    content = await file.read(services.max_upload_bytes + 1)
    return await services.documents.upload(
        principal,
        DocumentUpload(
            workspace_id=workspace_id,
            filename=file.filename or "",
            content_type=file.content_type or "application/octet-stream",
            content=content,
            visibility=visibility,
            acl_groups=tuple(acl_groups or []),
            idempotency_key=idempotency_key,
        ),
    )


@router.get("/{document_id}", response_model=DocumentStatusResponse)
async def document_status(
    document_id: str,
    workspace_id: str,
    principal: Annotated[Principal, Depends(get_principal)],
    services: Annotated[ApiServices, Depends(get_api_services)],
) -> DocumentStatusResponse:
    return await services.documents.status(principal, workspace_id, document_id)


@router.delete("/{document_id}", response_model=DocumentDeleteResponse)
async def delete_document(
    document_id: str,
    workspace_id: str,
    principal: Annotated[Principal, Depends(get_principal)],
    services: Annotated[ApiServices, Depends(get_api_services)],
) -> DocumentDeleteResponse:
    return await services.documents.delete(principal, workspace_id, document_id)
