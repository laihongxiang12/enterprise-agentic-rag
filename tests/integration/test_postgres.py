"""Real PostgreSQL tenant/ACL integration path; skipped without explicit DSN."""

import os
from uuid import uuid4

import pytest

from app.schemas.documents import DocumentMeta, Principal

pytestmark = pytest.mark.integration


def ready_document(
    *, tenant_id: str, document_id: str, groups: list[str], acl_version: int = 1
) -> DocumentMeta:
    return DocumentMeta(
        tenant_id=tenant_id,
        workspace_id="hr",
        document_id=document_id,
        version=1,
        title=f"Policy {document_id}",
        source_uri=f"fixture:///{document_id}.md",
        source_type="markdown",
        acl_groups=groups,
        content_hash="b" * 64,
        acl_version=acl_version,
        index_version=f"index-{tenant_id}-{document_id}-a{acl_version}",
        status="ready",
    )


@pytest.mark.asyncio
async def test_postgres_tenant_workspace_acl_and_version_update() -> None:
    database_url = os.getenv("TEST_DATABASE_URL")
    if database_url is None:
        pytest.skip("TEST_DATABASE_URL is not configured for real PostgreSQL integration")

    pytest.importorskip("psycopg", exc_type=ImportError)
    from psycopg import AsyncConnection

    from app.stores.postgres import PostgresMetadataStore, apply_migrations

    await apply_migrations(database_url)
    store = PostgresMetadataStore(database_url)
    suffix = uuid4().hex[:12]
    tenant_a = f"test-a-{suffix}"
    tenant_b = f"test-b-{suffix}"
    document_id = f"policy-{suffix}"
    tenant_b_only_document_id = f"finance-{suffix}"
    try:
        for tenant_id in (tenant_a, tenant_b):
            await store.put_tenant(tenant_id, tenant_id)
            await store.put_workspace(tenant_id, "hr", "HR")
        await store.put_workspace_membership(tenant_a, "hr", "alice")
        await store.put_workspace_membership(tenant_a, "hr", "charlie")
        await store.put_workspace_membership(tenant_b, "hr", "bob")
        await store.put_document(
            ready_document(tenant_id=tenant_a, document_id=document_id, groups=["hr"])
        )
        await store.put_document(
            ready_document(tenant_id=tenant_b, document_id=document_id, groups=["finance"])
        )
        await store.put_document(
            ready_document(
                tenant_id=tenant_b,
                document_id=tenant_b_only_document_id,
                groups=["hr"],
            )
        )

        alice = Principal(user_id="alice", tenant_id=tenant_a, group_ids=["hr"])
        bob = Principal(user_id="bob", tenant_id=tenant_b, group_ids=["finance"])
        assert (await store.get_authorized_document(alice, "hr", document_id)) is not None
        assert (await store.get_authorized_document(bob, "hr", document_id)) is not None
        assert not await store.delete_authorized_document(
            alice, "hr", tenant_b_only_document_id
        )
        assert (
            await store.get_document(tenant_b, "hr", tenant_b_only_document_id)
            is not None
        )
        scope = await store.authorization_scope(alice, "hr")
        assert scope.documents[0].index_version.startswith(f"index-{tenant_a}")

        await store.put_document(
            ready_document(
                tenant_id=tenant_a,
                document_id=document_id,
                groups=["finance"],
                acl_version=2,
            )
        )
        charlie = Principal(user_id="charlie", tenant_id=tenant_a, group_ids=["finance"])
        assert await store.get_authorized_document(alice, "hr", document_id) is None
        updated = await store.get_authorized_document(charlie, "hr", document_id)
        assert updated is not None and updated.acl_version == 2

        assert await store.delete_document(tenant_a, "hr", document_id)
        assert await store.get_authorized_document(charlie, "hr", document_id) is None
        assert await store.get_authorized_document(bob, "hr", document_id) is not None
    finally:
        connection = await AsyncConnection.connect(database_url)
        try:
            async with connection.transaction():
                await connection.execute(
                    "DELETE FROM enterprise_rag.tenants WHERE tenant_id = ANY(%s::text[])",
                    ([tenant_a, tenant_b],),
                )
        finally:
            await connection.close()
