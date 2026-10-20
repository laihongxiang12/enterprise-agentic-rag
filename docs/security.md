# Authentication, authorization, and retrieval scope

Stage 05 implements a local development authentication boundary, not enterprise SSO.

## Development authentication

`DEV_AUTH_ENABLED=true` is accepted only when all of the following hold:

- `APP_ENV=local`;
- `APP_MODE` is not `real`;
- `BIND_HOST` is `127.0.0.1`, `::1`, or `localhost`;
- `DEV_AUTH_SECRET` has at least 32 characters.

Tokens are short-lived HS256 JWTs with fixed issuer, audience, required expiry, and an algorithm
allowlist. Identity claims are mapped to `Principal` only after signature verification. This is a
demo boundary and must not be described as OIDC, SSO, or production identity federation.

## Fail-closed document policy

A document is readable only when tenant and workspace membership match, its active version is
`ready`, and either:

- `visibility=tenant` was explicitly stored; or
- `visibility=restricted` and at least one verified principal group intersects its ACL groups.

An empty ACL on a restricted document is denied. Client request bodies never supply tenant, user,
groups, scopes, ACL version, or active index version.

## Trusted retrieval pre-filter scope

PostgreSQL returns an immutable `AuthorizationScope` with this logical shape:

```json
{
  "tenant_id": "tenant-a",
  "workspace_id": "hr",
  "group_ids": ["employees", "hr"],
  "documents": [
    {
      "document_id": "leave-policy",
      "document_version": 2,
      "acl_version": 3,
      "index_version": "chunks-v1-..."
    }
  ]
}
```

Later retrieval adapters must push `tenant_id`, `workspace_id`, `document_id`, and `index_version`
into their database-side filter before selecting candidates. Empty `documents` means no retrieval;
it must never fall back to an unfiltered search. Group IDs are retained for audit/debug context but
the allowlist, versions, and tenant scope come from trusted PostgreSQL state.

## Real PostgreSQL integration

The repository Compose service binds PostgreSQL only to loopback. Copy `.env.example`
to ignored `.env`, replace its password, and run:

```bash
docker compose up -d postgres
docker compose ps postgres
uv run pytest tests/integration/test_postgres.py -m integration -q
```

`psycopg[binary]` supplies its own compatible `libpq` runtime inside the Python
environment. Without `TEST_DATABASE_URL`, the test is skipped and does not claim
PostgreSQL passed.

The test applies packaged migrations, creates uniquely named tenants, verifies cross-tenant and
group isolation plus ACL version changes, and removes only those test tenants afterward.

## Stage 17 security hardening

The application uses programmatic boundaries rather than relying on a model instruction:

- bearer tokens are length-bounded and signed identity claims reject control characters,
  oversized values, and oversized group/scope collections;
- the immutable default tool registry contains only `search_docs`; `web`, `sql`, `shell`,
  `python`, and filesystem-write tools cannot be registered or enabled by a forged scope;
- Milvus receives tenant, workspace, document, active document-version, and index-version
  constraints before search, and retrieval nodes validate returned evidence again;
- parent expansion and citations validate tenant/workspace/document/version/index, ACL visibility,
  and parent-child lineage;
- retrieval cache keys bind the complete authorization/version scope, and every cache hit is
  revalidated. A poisoned or stale entry is discarded, while an out-of-scope backend response
  fails closed and is never cached;
- checkpoint thread identifiers are derived from verified principal, group, workspace, and
  session scope rather than a caller-global session ID;
- free-form audit `details` are discarded. Stable IDs, action/type, and outcome remain in their
  dedicated columns; query text, document content, bearer tokens, prompts, and API keys are not
  persisted there;
- no trace-export HTTP route is currently implemented. `/v1/traces/*` therefore passes the
  authentication boundary and returns `404`; in-memory spans are for local tests/debug only.
  A future trace endpoint must add an explicit admin scope and redaction review before release.

External document text is serialized below `untrusted_evidence[*].content` in a JSON user message.
It cannot modify citation identity fields, identity, retrieval scope, or the tool registry. The
generator receives no executable tool definitions. The regression corpus contains 26 distinct,
harmless prompt-injection, identity-impersonation, cross-tenant, SQL, shell, forged-citation, and
malicious-document strings.

## Reproduce the negative tests

From WSL in the project directory:

```bash
uv run ruff check app/core/security.py app/stores/postgres.py \
  app/services/query_service.py app/tools tests/security
uv run mypy app
uv run pytest tests/security -q
uv run pytest -q
uv run pytest tests/integration/test_postgres.py \
  tests/integration/test_milvus.py tests/integration/test_checkpoint.py -m integration -q
```

Stage 17 results on 2026-10-18:

- security regressions: `52 passed`;
- full suite: `237 passed`, with one upstream Redis checkpoint deprecation warning;
- real PostgreSQL + Milvus + Redis checkpoint selection: `4 passed`, with the same warning;
- type checking: `Success: no issues found in 64 source files`;
- cross-tenant data returned by the negative tests: `0`.

The security suite intentionally does not call DeepSeek. These tests prove server-side identity,
pre-filter, cache, citation, checkpoint, audit, and tool-registry invariants; a paid model response
cannot prove those invariants and is not allowed to grant authority. Real DeepSeek configuration
is therefore neither required nor silently skipped for this stage.

Stage 21 re-ran this suite on 2026-10-20: `52 passed`; the full suite reported
`255 passed` and the selected real PostgreSQL/Milvus/Redis integration set reported
`5 passed` with one upstream Redis checkpoint deprecation warning.
