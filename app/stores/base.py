"""Narrow persistence contracts for later concrete adapters."""

from typing import Protocol, runtime_checkable

from app.schemas.documents import DocumentMeta


@runtime_checkable
class MetadataStore(Protocol):
    async def get_document(
        self,
        tenant_id: str,
        document_id: str,
    ) -> DocumentMeta | None: ...

    async def put_document(self, document: DocumentMeta) -> None: ...

    async def delete_document(self, tenant_id: str, document_id: str) -> None: ...


@runtime_checkable
class CacheStore(Protocol):
    async def get(self, key: str) -> bytes | None: ...

    async def set(self, key: str, value: bytes, ttl_seconds: int) -> None: ...

    async def invalidate(self, key_prefix: str) -> int: ...
