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

Set an isolated test database and run:

```bash
export TEST_DATABASE_URL='postgresql://user:password@127.0.0.1:5432/enterprise_rag_test'
uv run pytest tests/integration/test_postgres.py -m integration -q
```

The test applies packaged migrations, creates uniquely named tenants, verifies cross-tenant and
group isolation plus ACL version changes, and removes only those test tenants afterward.
