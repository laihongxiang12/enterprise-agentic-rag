"""Tenant-scoped in-memory metadata adapter for authorization unit tests."""

from app.core.security import BasicPolicyEngine
from app.schemas.documents import DocumentMeta, Principal
from app.schemas.retrieval import AuthorizationScope, AuthorizedDocument


class MemoryAuthorizationMetadataStore:
    """Fake store with the same fail-closed scope boundary as PostgreSQL."""

    def __init__(self) -> None:
        self.memberships: set[tuple[str, str, str]] = set()
        self.documents: dict[tuple[str, str, str], DocumentMeta] = {}
        self._policy = BasicPolicyEngine()

    async def put_workspace_membership(
        self, tenant_id: str, workspace_id: str, user_id: str
    ) -> None:
        self.memberships.add((tenant_id, workspace_id, user_id))

    async def put_document(self, document: DocumentMeta) -> None:
        key = (document.tenant_id, document.workspace_id, document.document_id)
        self.documents[key] = document.model_copy(deep=True)

    async def get_document(
        self, tenant_id: str, workspace_id: str, document_id: str
    ) -> DocumentMeta | None:
        document = self.documents.get((tenant_id, workspace_id, document_id))
        return None if document is None else document.model_copy(deep=True)

    async def authorization_scope(
        self, principal: Principal, workspace_id: str
    ) -> AuthorizationScope:
        workspace_authorized = (
            principal.tenant_id,
            workspace_id,
            principal.user_id,
        ) in self.memberships
        allowed: list[AuthorizedDocument] = []
        for document in self.documents.values():
            if document.workspace_id != workspace_id:
                continue
            if self._policy.authorize_read(
                principal,
                document,
                workspace_authorized=workspace_authorized,
            ):
                allowed.append(
                    AuthorizedDocument(
                        document_id=document.document_id,
                        index_version=document.index_version,
                        document_version=document.version,
                        acl_version=document.acl_version,
                    )
                )
        return AuthorizationScope(
            tenant_id=principal.tenant_id,
            workspace_id=workspace_id,
            group_ids=frozenset(principal.group_ids),
            documents=tuple(sorted(allowed, key=lambda item: item.document_id)),
        )

    async def get_authorized_document(
        self, principal: Principal, workspace_id: str, document_id: str
    ) -> DocumentMeta | None:
        scope = await self.authorization_scope(principal, workspace_id)
        if document_id not in {item.document_id for item in scope.documents}:
            return None
        return await self.get_document(principal.tenant_id, workspace_id, document_id)

    async def delete_authorized_document(
        self, principal: Principal, workspace_id: str, document_id: str
    ) -> bool:
        if await self.get_authorized_document(principal, workspace_id, document_id) is None:
            return False
        return await self.delete_document(principal.tenant_id, workspace_id, document_id)

    async def delete_document(
        self, tenant_id: str, workspace_id: str, document_id: str
    ) -> bool:
        key = (tenant_id, workspace_id, document_id)
        return self.documents.pop(key, None) is not None
