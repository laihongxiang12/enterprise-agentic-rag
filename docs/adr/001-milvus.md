# ADR 001: Milvus 2.5 authorized vector adapter

- Status: Accepted
- Date: 2026-10-17

## Decision

Use Milvus `2.5.27` with PyMilvus `2.5.18`. Milvus 2.5.27 is the current
security release in the 2.5 line and fixes a critical metrics-port authentication
bypass. The official compatibility table pairs the 2.5 server line with the
2.5 PyMilvus line.

References:

- https://milvus.io/docs/v2.5.x/release_notes.md
- https://milvus.io/api-reference/pymilvus/v2.5.x/About.md
- https://milvus.io/docs/v2.5.x/filtering-templating.md
- https://milvus.io/docs/v2.5.x/array-operators.md

## Security boundary

The collection uses an explicit schema and disables dynamic fields. Every dense
search sends tenant, workspace, PostgreSQL-derived document allowlist, active
index-version allowlist, exact document/document-version/index-version bindings,
ready state, visibility, and group ACL predicates to
Milvus in the search request. Values use `filter_params` placeholders rather
than expression string interpolation. No result is fetched from an unfiltered
collection and then filtered in Python.

The document/index allowlists remain the primary authorization decision from
trusted PostgreSQL. Milvus visibility and ARRAY group predicates are a
defense-in-depth copy that prevents a permissive document allowlist from exposing
a restricted chunk. Empty document/index allowlists or missing exact version
bindings return no results. Independent IN lists alone would admit obsolete
versions when an index label is shared by another document.

The local Compose deployment exposes only gRPC on `127.0.0.1:19530`. etcd,
MinIO, and Milvus port 9091 stay inside the Compose network. This loopback setup
does not claim production authentication, TLS, backup, or high availability.
The storage image is the `milvusdb/minio` build used by Milvus's maintained
standalone Compose because the former `minio/minio` Docker Hub repository is no
longer available.

## Schema and lifecycle

The primary key is SHA-256 over tenant, workspace, document, document version,
index version, and chunk ID. This permits old and new index versions to coexist.
Dense vectors use COSINE/AUTOINDEX. Scalar fields used by authorization have
INVERTED indexes. Upserted rows are marked ready; search additionally requires
the active index version from PostgreSQL. Delete requires the complete tenant,
workspace, document, document-version, and index-version scope.

Native sparse/BM25 fields and hybrid search are deliberately deferred to Stage
08. Stage 07's lexical method raises `NotImplementedError` rather than silently
falling back to an unsafe or fake production path. Existing FAISS/BM25 examples
remain unchanged as offline baselines.

Each adapter owns a unique PyMilvus connection alias. PyMilvus clients otherwise
share a URI-based connection, so closing one client can disconnect another live
adapter. The unique alias makes adapter lifecycle and cleanup independent.
