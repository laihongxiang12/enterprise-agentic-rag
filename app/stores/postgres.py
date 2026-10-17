"""Tenant-scoped PostgreSQL metadata, authorization, job, and audit adapter."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from psycopg import AsyncConnection
from psycopg.rows import DictRow, dict_row

from app.schemas.documents import DocumentMeta, Principal
from app.schemas.retrieval import AuthorizationScope, AuthorizedDocument
from app.stores.models import AuditEvent, IngestJob

_MIGRATIONS = Path(__file__).with_name("migrations")


@asynccontextmanager
async def _connect(database_url: str) -> AsyncIterator[AsyncConnection[DictRow]]:
    connection = await AsyncConnection.connect(database_url, row_factory=dict_row)
    try:
        yield connection
    finally:
        await connection.close()


async def apply_migrations(database_url: str) -> list[str]:
    """Apply packaged SQL migrations once under a PostgreSQL advisory lock."""
    applied: list[str] = []
    async with _connect(database_url) as connection, connection.transaction():
        await connection.execute("CREATE SCHEMA IF NOT EXISTS enterprise_rag")
        await connection.execute(
            """
            CREATE TABLE IF NOT EXISTS enterprise_rag.schema_migrations (
                version TEXT PRIMARY KEY,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        await connection.execute(
            "SELECT pg_advisory_xact_lock(hashtext(%s))",
            ("enterprise-agentic-rag-migrations",),
        )
        cursor = await connection.execute(
            "SELECT version FROM enterprise_rag.schema_migrations"
        )
        completed = {str(row["version"]) for row in await cursor.fetchall()}
        for path in sorted(_MIGRATIONS.glob("*.sql")):
            if path.stem in completed:
                continue
            await connection.execute(path.read_text(encoding="utf-8"))
            await connection.execute(
                "INSERT INTO enterprise_rag.schema_migrations (version) VALUES (%s)",
                (path.stem,),
            )
            applied.append(path.stem)
    return applied


class PostgresMetadataStore:
    """No method exposes an unscoped document or audit-table read."""

    def __init__(self, database_url: str) -> None:
        if not database_url.strip():
            raise ValueError("database_url must not be empty")
        self._database_url = database_url

    async def put_tenant(self, tenant_id: str, display_name: str) -> None:
        async with _connect(self._database_url) as connection, connection.transaction():
            await connection.execute(
                """
                INSERT INTO enterprise_rag.tenants (tenant_id, display_name)
                VALUES (%s, %s)
                ON CONFLICT (tenant_id) DO UPDATE SET display_name = EXCLUDED.display_name
                """,
                (tenant_id, display_name),
            )

    async def put_workspace(
        self, tenant_id: str, workspace_id: str, display_name: str
    ) -> None:
        async with _connect(self._database_url) as connection, connection.transaction():
            await connection.execute(
                """
                INSERT INTO enterprise_rag.workspaces (tenant_id, workspace_id, display_name)
                VALUES (%s, %s, %s)
                ON CONFLICT (tenant_id, workspace_id)
                DO UPDATE SET display_name = EXCLUDED.display_name
                """,
                (tenant_id, workspace_id, display_name),
            )

    async def put_workspace_membership(
        self, tenant_id: str, workspace_id: str, user_id: str
    ) -> None:
        async with _connect(self._database_url) as connection, connection.transaction():
            await connection.execute(
                """
                INSERT INTO enterprise_rag.workspace_memberships
                    (tenant_id, workspace_id, user_id)
                VALUES (%s, %s, %s)
                ON CONFLICT DO NOTHING
                """,
                (tenant_id, workspace_id, user_id),
            )

    async def put_document(self, document: DocumentMeta) -> None:
        """Upsert one immutable content/ACL version and optionally activate it."""
        async with _connect(self._database_url) as connection, connection.transaction():
            await connection.execute(
                """
                INSERT INTO enterprise_rag.documents (
                    tenant_id, workspace_id, document_id, title, source_uri,
                    source_type, visibility, active_version, active_acl_version,
                    active_index_version, deleted_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s,
                    CASE WHEN %s = 'ready' THEN %s ELSE NULL END,
                    CASE WHEN %s = 'ready' THEN %s ELSE NULL END,
                    CASE WHEN %s = 'ready' THEN %s ELSE NULL END,
                    NULL
                )
                ON CONFLICT (tenant_id, workspace_id, document_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    source_uri = EXCLUDED.source_uri,
                    source_type = EXCLUDED.source_type,
                    visibility = EXCLUDED.visibility,
                    active_version = CASE WHEN %s = 'ready' THEN %s
                        ELSE enterprise_rag.documents.active_version END,
                    active_acl_version = CASE WHEN %s = 'ready' THEN %s
                        ELSE enterprise_rag.documents.active_acl_version END,
                    active_index_version = CASE WHEN %s = 'ready' THEN %s
                        ELSE enterprise_rag.documents.active_index_version END,
                    deleted_at = NULL,
                    updated_at = now()
                """,
                (
                    document.tenant_id,
                    document.workspace_id,
                    document.document_id,
                    document.title,
                    document.source_uri,
                    document.source_type,
                    document.visibility,
                    document.status,
                    document.version,
                    document.status,
                    document.acl_version,
                    document.status,
                    document.index_version,
                    document.status,
                    document.version,
                    document.status,
                    document.acl_version,
                    document.status,
                    document.index_version,
                ),
            )
            await connection.execute(
                """
                INSERT INTO enterprise_rag.document_versions (
                    tenant_id, workspace_id, document_id, version, acl_version,
                    content_hash, index_version, status
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (tenant_id, workspace_id, document_id, version, acl_version)
                DO UPDATE SET content_hash = EXCLUDED.content_hash,
                    index_version = EXCLUDED.index_version,
                    status = EXCLUDED.status
                """,
                (
                    document.tenant_id,
                    document.workspace_id,
                    document.document_id,
                    document.version,
                    document.acl_version,
                    document.content_hash,
                    document.index_version,
                    document.status,
                ),
            )
            await connection.execute(
                """
                DELETE FROM enterprise_rag.document_acl_groups
                WHERE tenant_id = %s AND workspace_id = %s AND document_id = %s
                    AND version = %s AND acl_version = %s
                """,
                (
                    document.tenant_id,
                    document.workspace_id,
                    document.document_id,
                    document.version,
                    document.acl_version,
                ),
            )
            if document.acl_groups:
                async with connection.cursor() as cursor:
                    await cursor.executemany(
                        """
                        INSERT INTO enterprise_rag.document_acl_groups (
                            tenant_id, workspace_id, document_id, version, acl_version, group_id
                        ) VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        [
                            (
                                document.tenant_id,
                                document.workspace_id,
                                document.document_id,
                                document.version,
                                document.acl_version,
                                group_id,
                            )
                            for group_id in sorted(set(document.acl_groups))
                        ],
                    )

    async def get_document(
        self, tenant_id: str, workspace_id: str, document_id: str
    ) -> DocumentMeta | None:
        async with _connect(self._database_url) as connection:
            cursor = await connection.execute(
                _ACTIVE_DOCUMENT_SQL,
                (tenant_id, workspace_id, document_id),
            )
            row = await cursor.fetchone()
        return None if row is None else DocumentMeta.model_validate(row)

    async def authorization_scope(
        self, principal: Principal, workspace_id: str
    ) -> AuthorizationScope:
        """Compute a bounded allowlist in PostgreSQL before vector retrieval."""
        async with _connect(self._database_url) as connection:
            cursor = await connection.execute(
                """
                SELECT d.document_id, d.active_index_version AS index_version,
                    d.active_version AS document_version,
                    d.active_acl_version AS acl_version
                FROM enterprise_rag.workspace_memberships AS membership
                JOIN enterprise_rag.documents AS d
                    ON d.tenant_id = membership.tenant_id
                    AND d.workspace_id = membership.workspace_id
                JOIN enterprise_rag.document_versions AS version
                    ON version.tenant_id = d.tenant_id
                    AND version.workspace_id = d.workspace_id
                    AND version.document_id = d.document_id
                    AND version.version = d.active_version
                    AND version.acl_version = d.active_acl_version
                WHERE membership.tenant_id = %s
                    AND membership.workspace_id = %s
                    AND membership.user_id = %s
                    AND d.deleted_at IS NULL
                    AND version.status = 'ready'
                    AND (
                        d.visibility = 'tenant'
                        OR EXISTS (
                            SELECT 1 FROM enterprise_rag.document_acl_groups AS acl
                            WHERE acl.tenant_id = d.tenant_id
                                AND acl.workspace_id = d.workspace_id
                                AND acl.document_id = d.document_id
                                AND acl.version = d.active_version
                                AND acl.acl_version = d.active_acl_version
                                AND acl.group_id = ANY(%s::text[])
                        )
                    )
                ORDER BY d.document_id
                """,
                (
                    principal.tenant_id,
                    workspace_id,
                    principal.user_id,
                    principal.group_ids,
                ),
            )
            rows = await cursor.fetchall()
        return AuthorizationScope(
            tenant_id=principal.tenant_id,
            workspace_id=workspace_id,
            group_ids=frozenset(principal.group_ids),
            documents=tuple(AuthorizedDocument.model_validate(row) for row in rows),
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
        """Delete only when the verified principal can currently read the document."""
        if await self.get_authorized_document(principal, workspace_id, document_id) is None:
            return False
        return await self.delete_document(principal.tenant_id, workspace_id, document_id)

    async def delete_document(
        self, tenant_id: str, workspace_id: str, document_id: str
    ) -> bool:
        async with _connect(self._database_url) as connection, connection.transaction():
            cursor = await connection.execute(
                """
                UPDATE enterprise_rag.documents
                SET deleted_at = now(), updated_at = now()
                WHERE tenant_id = %s AND workspace_id = %s AND document_id = %s
                    AND deleted_at IS NULL
                RETURNING active_version, active_acl_version
                """,
                (tenant_id, workspace_id, document_id),
            )
            row = await cursor.fetchone()
            if row is None:
                return False
            await connection.execute(
                """
                UPDATE enterprise_rag.document_versions SET status = 'deleted'
                WHERE tenant_id = %s AND workspace_id = %s AND document_id = %s
                    AND version = %s AND acl_version = %s
                """,
                (
                    tenant_id,
                    workspace_id,
                    document_id,
                    row["active_version"],
                    row["active_acl_version"],
                ),
            )
            return True

    async def put_ingest_job(self, job: IngestJob) -> None:
        async with _connect(self._database_url) as connection, connection.transaction():
            await connection.execute(
                """
                INSERT INTO enterprise_rag.ingest_jobs (
                    tenant_id, workspace_id, job_id, document_id, status, error_code
                ) VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (tenant_id, job_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    error_code = EXCLUDED.error_code,
                    updated_at = now()
                """,
                (
                    job.tenant_id,
                    job.workspace_id,
                    job.job_id,
                    job.document_id,
                    job.status,
                    job.error_code,
                ),
            )

    async def append_audit(self, event: AuditEvent) -> None:
        async with _connect(self._database_url) as connection, connection.transaction():
            await connection.execute(
                """
                INSERT INTO enterprise_rag.audit_events (
                    tenant_id, event_id, actor_user_id, action, resource_type,
                    resource_id, outcome, trace_id, details
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                """,
                (
                    event.tenant_id,
                    event.event_id,
                    event.actor_user_id,
                    event.action,
                    event.resource_type,
                    event.resource_id,
                    event.outcome,
                    event.trace_id,
                    json.dumps(event.details, ensure_ascii=False),
                ),
            )


_ACTIVE_DOCUMENT_SQL = """
    SELECT d.tenant_id, d.workspace_id, d.document_id, d.active_version AS version,
        d.title, d.source_uri, d.source_type, d.visibility,
        version.content_hash, d.active_acl_version AS acl_version,
        d.active_index_version AS index_version, version.status,
        COALESCE(
            ARRAY_AGG(acl.group_id ORDER BY acl.group_id)
                FILTER (WHERE acl.group_id IS NOT NULL),
            ARRAY[]::text[]
        ) AS acl_groups
    FROM enterprise_rag.documents AS d
    JOIN enterprise_rag.document_versions AS version
        ON version.tenant_id = d.tenant_id
        AND version.workspace_id = d.workspace_id
        AND version.document_id = d.document_id
        AND version.version = d.active_version
        AND version.acl_version = d.active_acl_version
    LEFT JOIN enterprise_rag.document_acl_groups AS acl
        ON acl.tenant_id = version.tenant_id
        AND acl.workspace_id = version.workspace_id
        AND acl.document_id = version.document_id
        AND acl.version = version.version
        AND acl.acl_version = version.acl_version
    WHERE d.tenant_id = %s AND d.workspace_id = %s AND d.document_id = %s
        AND d.deleted_at IS NULL
    GROUP BY d.tenant_id, d.workspace_id, d.document_id, d.active_version,
        d.title, d.source_uri, d.source_type, d.visibility, version.content_hash,
        d.active_acl_version, d.active_index_version, version.status
"""
