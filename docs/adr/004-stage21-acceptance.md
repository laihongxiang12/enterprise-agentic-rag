# ADR 004: Stage 21 independent acceptance

- Status: Accepted as an audit record
- Date: 2026-10-20
- Release verdict: **CONDITIONAL — portfolio/demo ready, not production ready**

## Decision

The repository is accepted as a reproducible engineering portfolio and local Fake Demo. It is not
accepted as a production real-mode service. The verdict is intentionally narrower than “all
features complete”: real adapters are independently integration-tested, but no API composition
root wires them into one `APP_MODE=real` request path.

Use these labels throughout project claims:

- `VERIFIED`: code path and its relevant test passed in this acceptance run;
- `PARTIAL`: a real component exists, but the advertised end-to-end path is not assembled;
- `NOT_IMPLEMENTED`: no supported endpoint or composition exists;
- `NOT_MEASURED`: no valid measurement was performed; never substitute fixture data.

## Acceptance matrix

| Area | Verdict | Evidence and boundary |
|---|---|---|
| Loader, stable parent/child chunks, versioned ingestion | VERIFIED | Unit and ingestion tests are included in the 178-unit pass. |
| PostgreSQL metadata and ACL authorization scope | VERIFIED | Real PostgreSQL integration passed; server builds the allowlist and active versions. |
| Milvus Dense + native BM25 + RRF | VERIFIED | Two real Milvus integration paths passed with database-side tenant/workspace/document/version filters. |
| Redis cache, rate-limit seam, checkpoint isolation | VERIFIED | Cache/security tests and real Redis checkpoint integration passed. |
| LangGraph routing, retry, multi-hop, budgets and deadline | VERIFIED | Deterministic graph tests verify bounded termination and fail-closed errors. |
| Grounded generation and citation verification | VERIFIED | Deterministic provider tests reject missing, forged, cross-scope and unverified citations. |
| ACL before retrieval and after retrieval | VERIFIED | PostgreSQL scope, Milvus expression, retrieval node, parent expansion, grader and citation stages revalidate scope. |
| Tool-policy bypass resistance | VERIFIED | Immutable allowlist exposes only `search_docs`; forged scopes cannot enable shell, Python, SQL, web or writes. |
| Cache/index version leakage resistance | VERIFIED | Cache key binds the complete authorization/version scope; hits are revalidated and poisoned entries discarded. |
| SSE contract | VERIFIED | Event order and terminal exclusivity pass, but `delta` is explicitly simulated final-answer segmentation, not model token streaming. |
| Browser Compose Demo | VERIFIED | Non-root image rebuilt; six services became healthy; seed and success/rewrite/unauthorized smoke cases passed. |
| PostgreSQL/Milvus/Redis in the browser query path | PARTIAL | Containers are healthy, but `DemoQueryService` does not consume them for answers. |
| `APP_MODE=real` API composition root | NOT_IMPLEMENTED | Real mode does not fall back to Fake and `/health/ready` returns 503 until real services/probes are assembled. |
| Production SSO and durable ingestion worker | NOT_IMPLEMENTED | Only local demo/dev seams exist; deployment owner must supply these integrations. |
| `/v1/evaluations` and `/v1/traces/{trace_id}` | NOT_IMPLEMENTED | No routes exist. `/metrics` and in-process test traces do exist. |
| Versioned source documents under `evals/datasets/documents/` | NOT_IMPLEMENTED | The committed reproducible source is currently the 25-record `corpus.jsonl`; original source-document files are absent. |
| Real DeepSeek evaluation | NOT_MEASURED | API key presence was checked without disclosure. The real eval adapter intentionally fails closed and no paid request was made. |
| Production latency and cost | NOT_MEASURED | Fixture latency is sub-millisecond in-process work and cannot represent networked production. |

## Confirmed defects fixed during acceptance

1. The graph `direct` node returned `generation=deferred_to_stage_14`, which became an empty API
   answer. It now returns an explicit safe, non-enterprise answer with no citations and a completed
   finish reason; a regression test rejects the old placeholder.
2. `/health/ready` was missing and Compose checked only liveness. The new endpoint reports the
   assembled local Fake runtime as ready and fails closed with 503 for unassembled local/real
   modes. Compose now uses readiness.
3. The evidence grader repeated version checks but omitted the visibility/group predicate. It now
   uses the same authorization function as retrieval, preserving defense in depth.
4. Architecture and security documents contained stage-era status text. They now distinguish
   current implementation, demo behavior, missing endpoints and unmeasured real-model results.

## Reproduction record

Environment: WSL2, Python 3.12, uv-managed environment, Docker Compose. Commands were run from the
product root with ignored local `TEST_*` variables loaded; secrets were not printed.

| Gate | Result |
|---|---:|
| `uv lock --check` | PASS |
| `uv run ruff check .` | PASS |
| `uv run mypy app evals scripts` | PASS — 80 source files |
| `uv run pytest tests/unit -q` | PASS — 178 |
| `uv run pytest tests/security -q` | PASS — 52 |
| selected real PostgreSQL/Milvus/Redis integrations | PASS — 5, 1 upstream warning |
| `uv run pytest -q` | PASS — 255, 1 upstream warning |
| full offline evaluation | PASS — 25 corpus, 108 golden, 20 injection |
| Compose build/wait + seed + smoke | PASS — success, rewrite, unauthorized |
| `GET /health/ready` in Fake Demo | PASS — HTTP 200 `ready` |

The warning is a dependency deprecation notice from `langgraph-checkpoint-redis`; it is not a test
failure. The full offline D baseline achieved 1.0 on its committed scripted fixtures. That is a
workflow upper bound, not a claim about autonomous DeepSeek quality.

## Security-focused review notes

- Identity is derived from verified credentials; request bodies cannot select tenant or user.
- Milvus filters before candidate return. Empty allowlists fail closed.
- Every loop has retry, tool-call, round and deadline budgets.
- Evidence IDs must be present in the authorized retrieval set before becoming citations.
- Checkpoint and cache identities include principal and authorization/version scope.
- Query text, prompts, document content, bearer tokens and API keys are excluded from structured
  observability fields.
- The HTTP Demo graph display is a deterministic fixture. It must not be presented as a captured
  production trace.

## Required work before production acceptance

1. Build a real composition root that wires production authentication, PostgreSQL authorization,
   Milvus retrieval, Redis controls/checkpoint, the Agent application service and DeepSeek.
2. Add dependency-specific readiness probes and make real mode return 200 only after all required
   dependencies are verified.
3. Add a durable ingestion queue/worker and operational recovery procedures.
4. Implement authorized evaluation/trace APIs only if they are still product requirements.
5. Run an explicitly approved, versioned real-model evaluation and report provider usage/cost plus
   networked p50/p95. Do not reuse offline fixture latency.

## Resume claim boundary

The following bullets describe repository capabilities and measured results. Use first-person
ownership only after confirming which parts the candidate personally designed and implemented.

- Built and tested a tenant-aware Agentic RAG portfolio with PostgreSQL ACL metadata, Milvus
  hybrid retrieval, Redis isolation and bounded LangGraph workflows; the final WSL acceptance run
  passed 255 tests, including 52 security regressions and 5 real backend integrations.
- Designed fail-closed retrieval and citation controls that bind tenant, workspace, document,
  document version and index version; verified zero authorized-path acceptance of the 20 committed
  injection fixtures in the offline evaluation.
- Created a 25-chunk fictional policy corpus with 108 golden queries and 20 security cases, and
  documented a reproducible A/B/C/D ablation without representing deterministic fixture metrics as
  production model quality.
- Packaged a non-root six-service Compose Demo with readiness, idempotent seed and three-case smoke
  validation, while explicitly separating the Fake UI fixture from unimplemented real-mode wiring.
