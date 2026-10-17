"""Contracts and stable errors for document loading."""

from enum import StrEnum
from typing import Annotated

from pydantic import Field

from app.schemas.base import ContractModel

NonEmptyString = Annotated[str, Field(min_length=1)]
MetadataValue = str | int | bool | None


class PageBlock(ContractModel):
    """A source-local text block; PDF pages are one-based when known."""

    text: NonEmptyString
    page: int | None = Field(default=None, ge=1)
    heading_path: list[str] = Field(default_factory=list)


class LoadedDocument(ContractModel):
    """Canonical output shared by every supported loader."""

    text: NonEmptyString
    title: NonEmptyString
    source_uri: NonEmptyString
    page_blocks: list[PageBlock] = Field(min_length=1)
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)


class LoadErrorCode(StrEnum):
    """Machine-readable document load failure categories."""

    FILE_NOT_FOUND = "file_not_found"
    UNSAFE_PATH = "unsafe_path"
    UNSUPPORTED_TYPE = "unsupported_type"
    MIME_MISMATCH = "mime_mismatch"
    FILE_TOO_LARGE = "file_too_large"
    EMPTY_FILE = "empty_file"
    INVALID_ENCODING = "invalid_encoding"
    INVALID_DOCUMENT = "invalid_document"
    ENCRYPTED_PDF = "encrypted_pdf"
    NO_EXTRACTABLE_TEXT = "no_extractable_text"


class DocumentLoadError(ValueError):
    """Expected parse failure with a safe, stable error code."""

    def __init__(self, code: LoadErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
