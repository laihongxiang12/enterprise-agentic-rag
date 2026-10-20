# Read-only MCP server

Stage 22 uses the official Model Context Protocol Python SDK `mcp==2.3.0`. The server exposes
exactly two read-only tools:

| Tool | Model-visible arguments | Result |
|---|---|---|
| `search_docs` | `query`, optional `limit` (1–20) | Authorized chunk IDs, bounded snippets, source metadata and scores |
| `get_doc` | `document_id` | Authorized document metadata; no arbitrary file read and no document body dump |

There are no shell, Python, filesystem-write, SQL, Web, upload, delete or administration tools.
Unknown tool arguments are rejected with MCP `Invalid Params` before SDK function binding.

## Trust and authorization boundary

The model never supplies `tenant_id`, `user_id`, groups, scopes or workspace. A verified
`Principal`, one workspace and injected service dependencies are fixed in
`MCPConnectionContext` before stdio starts. Each tool then:

1. checks the immutable `ToolPolicy` for its exact `tools:*` scope;
2. invokes the shared request limiter;
3. derives an `AuthorizationScope` from the metadata store and verified principal;
4. sends its `AuthFilter` into the existing Retriever before candidate selection;
5. revalidates every returned chunk against tenant/workspace/document/version/index/group scope;
6. stops at a configured timeout and maps dependency exceptions to stable error codes.

`get_doc` uses `MetadataStore.get_authorized_document`. Missing and unauthorized IDs both return
`not_found`, preventing cross-tenant existence disclosure.

## Local stdio demonstration

From the product root in WSL:

```bash
uv sync --extra dev
uv run python -m scripts.mcp_client
```

The client creates an ephemeral signing secret, issues a short-lived principal with only
`tools:get_doc` and `tools:search_docs`, and explicitly passes the minimum environment to the child
process. The SDK performs a real stdio protocol negotiation, `tools/list`, `search_docs`, and
`get_doc`. The secret disappears with the processes and is never printed.

This demonstration uses an in-memory metadata store and Fake embedding/vector adapter over original
fictional Northstar data. It exercises real MCP framing, schema, policy, ACL pre-filter, Hybrid/RRF
and result validation, but it is not a PostgreSQL/Milvus production measurement and does not call
DeepSeek.

## Host configuration

A desktop MCP host can launch the same entry point:

```json
{
  "command": "/absolute/path/to/.venv/bin/python",
  "args": ["-m", "app.mcp.server"],
  "cwd": "/absolute/path/to/enterprise-agentic-rag",
  "env": {
    "DEV_AUTH_SECRET": "a-local-secret-of-at-least-32-characters",
    "MCP_DEV_BEARER_TOKEN": "a-short-lived-signed-token",
    "MCP_WORKSPACE_ID": "policies"
  }
}
```

Do not commit this host configuration with a real token. The server verifies the token before
constructing the MCP context. A malformed, expired, incorrectly signed or missing token aborts
startup rather than creating an anonymous principal.

## Transport limitation

The delivered CLI is stdio-only and intended for a trusted local host. In stdio, the process that
launches the server is the connection security boundary; there is no HTTP Authorization header.
The official SDK also supports Streamable HTTP, but publishing this server over HTTP requires a
production OAuth/OIDC token verifier, per-request principal construction, TLS and a real dependency
composition root. Reusing this server-wide stdio context across HTTP users would be unsafe and is
not supported.

## Tests

```bash
uv run pytest tests/integration/test_mcp.py -q
uv run pytest tests/security/test_mcp_scope.py -q
uv run pytest tests/security -q
uv run pytest -q
```

The protocol test uses the SDK's in-memory transport, which still performs real tool listing,
schema validation and invocation without opening a network port. The separate client command proves
the stdio subprocess path. Tests cover unknown arguments, unknown/dangerous tools, missing scopes,
cross-tenant document access, malicious retriever output, bounded timeout and sanitized errors.
