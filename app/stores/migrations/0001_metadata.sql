CREATE TABLE IF NOT EXISTS enterprise_rag.tenants (
    tenant_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS enterprise_rag.workspaces (
    tenant_id TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    display_name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, workspace_id),
    FOREIGN KEY (tenant_id) REFERENCES enterprise_rag.tenants (tenant_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS enterprise_rag.workspace_memberships (
    tenant_id TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, workspace_id, user_id),
    FOREIGN KEY (tenant_id, workspace_id)
        REFERENCES enterprise_rag.workspaces (tenant_id, workspace_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS enterprise_rag.documents (
    tenant_id TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    title TEXT NOT NULL,
    source_uri TEXT NOT NULL,
    source_type TEXT NOT NULL CHECK (source_type IN ('pdf', 'markdown', 'html')),
    visibility TEXT NOT NULL CHECK (visibility IN ('restricted', 'tenant')),
    active_version INTEGER,
    active_acl_version INTEGER,
    active_index_version TEXT,
    deleted_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, workspace_id, document_id),
    FOREIGN KEY (tenant_id, workspace_id)
        REFERENCES enterprise_rag.workspaces (tenant_id, workspace_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS enterprise_rag.document_versions (
    tenant_id TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    version INTEGER NOT NULL CHECK (version >= 1),
    acl_version INTEGER NOT NULL CHECK (acl_version >= 1),
    content_hash TEXT NOT NULL,
    index_version TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'processing', 'ready', 'failed', 'deleted')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, workspace_id, document_id, version, acl_version),
    UNIQUE (tenant_id, index_version),
    FOREIGN KEY (tenant_id, workspace_id, document_id)
        REFERENCES enterprise_rag.documents (tenant_id, workspace_id, document_id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS enterprise_rag.document_acl_groups (
    tenant_id TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    acl_version INTEGER NOT NULL,
    group_id TEXT NOT NULL,
    PRIMARY KEY (tenant_id, workspace_id, document_id, version, acl_version, group_id),
    FOREIGN KEY (tenant_id, workspace_id, document_id, version, acl_version)
        REFERENCES enterprise_rag.document_versions
            (tenant_id, workspace_id, document_id, version, acl_version)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS enterprise_rag.ingest_jobs (
    tenant_id TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'succeeded', 'failed')),
    error_code TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, job_id),
    FOREIGN KEY (tenant_id, workspace_id)
        REFERENCES enterprise_rag.workspaces (tenant_id, workspace_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS enterprise_rag.audit_events (
    tenant_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    actor_user_id TEXT NOT NULL,
    action TEXT NOT NULL,
    resource_type TEXT NOT NULL,
    resource_id TEXT NOT NULL,
    outcome TEXT NOT NULL CHECK (outcome IN ('allowed', 'denied', 'success', 'failure')),
    trace_id TEXT NOT NULL,
    details JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, event_id),
    FOREIGN KEY (tenant_id) REFERENCES enterprise_rag.tenants (tenant_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_document_acl_lookup
    ON enterprise_rag.document_acl_groups
    (tenant_id, workspace_id, document_id, version, acl_version, group_id);

CREATE INDEX IF NOT EXISTS idx_audit_tenant_created
    ON enterprise_rag.audit_events (tenant_id, created_at DESC);
