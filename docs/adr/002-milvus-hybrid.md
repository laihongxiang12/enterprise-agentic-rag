# ADR 002: Native Milvus BM25 and application-level RRF

- Status: Accepted
- Date: 2026-10-17

## Decision

Use the Milvus 2.5 native BM25 function with a `VARCHAR` source field, a
`SPARSE_FLOAT_VECTOR` function output, and a `SPARSE_INVERTED_INDEX`. Chinese
text uses the built-in Jieba analyzer. Query text is sent directly to Milvus;
the service does not train, persist, or synthesize sparse vectors itself.

Run dense and lexical searches as two independent, concurrent requests with
the exact same trusted server-side authorization expression. Defaults are
dense top 30, lexical top 30, RRF `k=60`, and fused top 20. The application
performs reciprocal-rank fusion so the offline baseline and production path
share one deterministic, testable implementation. Raw COSINE and BM25 scores
are not added together.

References:

- https://milvus.io/docs/v2.5.x/full-text-search.md
- https://milvus.io/docs/v2.5.x/jieba-tokenizer.md
- https://milvus.io/api-reference/pymilvus/v2.5.x/MilvusClient/Function/Function.md

## Authorization and version boundary

Both branches require tenant, workspace, document and active-index allowlists,
exact document/document-version/index-version bindings, ready state, visibility,
and group ACL predicates before scoring. Empty or incomplete metadata returns
no candidates and performs no search. Stable RRF deduplicates a chunk identity
and uses source order and identity as deterministic tie-breakers.

The eval export is an in-memory, JSON-serializable record containing query,
parameters, ranked branch results, and fused results. Export is explicit; the
retriever does not automatically write potentially sensitive content to disk.

## Collection migration

Milvus cannot add the BM25 function and sparse output field to the Stage 07
collection in place. The default collection therefore changes from
`enterprise_rag_chunks` to `enterprise_rag_hybrid_chunks`. This preserves the
old collection rather than deleting data. Re-ingestion into the new collection
is required before switching a real deployment. The old collection may be
removed only through an explicit, separately reviewed data-retention action.
