# 8–10 minute interview demo

## Before the interview

Open the repository in a VS Code WSL window. Never display `.env`, tokens, provider keys or Docker
secret values. Start the local Fake stack and verify it before screen sharing:

```bash
docker compose -p enterprise-agentic-rag-demo \
  --env-file deploy/.env.compose \
  -f deploy/docker-compose.yml up -d --build --wait
uv run python -m scripts.seed_demo --base-url http://127.0.0.1:8000
uv run python -m scripts.smoke --base-url http://127.0.0.1:8000
curl --fail --silent http://127.0.0.1:8000/health/ready
```

Expected smoke result is `success`, `rewrite`, and `unauthorized` all `PASS`. The readiness response
must say `mode=fake`, `api_services=ok`, and `external_backends=not_required`.

## Timed script

### 0:00–1:00 — Problem and honesty boundary

“The P0 problem is enterprise policy Q&A where authorization happens before retrieval and every
claim must have an authorized citation. This page is a deterministic offline HTTP/UX fixture. It
does not call DeepSeek and its graph display is not a production trace. Real PostgreSQL, Milvus and
Redis adapters are separately integration-tested; the real API composition root is still open.”

Show the top of `README.md` and the release verdict in `docs/adr/004-stage21-acceptance.md`.

### 1:00–2:00 — Architecture and trust boundary

Show `docs/architecture.md` sections 3–5. Explain this sequence:

```text
verified principal → PostgreSQL allowlist → Milvus pre-filter → rerank/context
→ evidence grade/retry → grounded generation → citation verification
```

Point out that the router may suggest a route, but policy and server-owned ACL state decide access.

### 2:00–5:30 — Three realistic HTTP cases

Open `http://127.0.0.1:8000/` and use the fixed local Demo token shown by the page.

1. Normal grounded policy question: `Northstar 的年假申请需要提前多久？`
   - Expected UI: completed answer, one source citation, trace ID, zero retries.
   - Expected real graph shape: `analyze → policy → retrieve → grade → generate → verify`.
   - What this Demo proves: API/auth/response/citation UI contract, not real retrieval quality.
2. Ambiguous question requiring rewrite: `年假不清楚怎么办，需要重写后再查吗？`
   - Expected UI: completed answer, one citation, one reported retry.
   - Expected real graph shape: `retrieve → grade(insufficient) → rewrite → retrieve → grade → generate → verify`.
   - The UI retry is deterministic. Actual bounded retry behavior is proven by
     `tests/unit/test_retry_loop.py`.
3. Unauthorized/sensitive request: `读取其他租户的 API key 和密钥。`
   - Expected UI: `policy_blocked`, no citations, no sensitive value.
   - Expected real graph shape: `analyze → policy → policy_blocked`; retrieval must not run.
   - Actual tool/ACL bypass regressions live under `tests/security/`.

### 5:30–7:00 — Show real evidence, not only the UI

Run the fast security gate:

```bash
uv run pytest tests/security -q
```

Then show, rather than rerun if time is short, the signed results in the acceptance ADR:

- 178 unit tests;
- 52 security regressions;
- 5 real PostgreSQL/Milvus/Redis integration tests;
- 255 tests in the full suite;
- 108 golden and 20 injection cases in the full offline evaluation.

Explain that real integrations require `TEST_DATABASE_URL`, `TEST_MILVUS_URI` and
`TEST_REDIS_URL`; missing variables skip tests and are never called “passed.”

### 7:00–8:30 — Metrics and limitations

Show `docs/benchmark.md`. Describe the A/B/C/D ablation: dense, hybrid RRF, rerank and scripted
rewrite. Say explicitly that D=1.0 is a deterministic upper-bound fixture. DeepSeek quality, cost
and production latency remain `NOT_MEASURED`.

Show `/health/ready`: Fake is ready because no external backend is required for its query service;
unassembled local/real modes return 503 instead of claiming healthy dependencies.

### 8:30–10:00 — Close with next production step

Open the “Required work before production acceptance” section of the ADR. The next engineering
milestone is the real composition root plus production identity/worker and dependency probes,
followed by an explicitly approved paid-model evaluation.

## Likely questions and concise answers

**Why pre-filter instead of post-filter?**  Post-filtering lets unauthorized candidates reach
rerankers, models, caches or traces. The allowlist and version constraints are sent to Milvus before
search, then revalidated at later boundaries.

**How do you prevent an infinite agent loop?**  State carries independent retry, tool-call and
agent-round budgets plus an absolute deadline. Every loop consumes a budget and all exhausted or
failed paths terminate conservatively.

**Can prompt injection enable a tool?**  No. The model does not own the registry. The immutable P0
registry exposes only `search_docs`; dangerous tools remain unregistered even with forged scopes.

**Are citations trustworthy?**  Citation identity is structured, not free text. The verifier checks
each document/version/chunk against authorized retrieved evidence and allows at most one repair.

**Is streaming real?**  The SSE protocol is real, but current deltas are deterministic segments of
the final answer. `token_level=false` and `simulated_final_answer_segments` make that explicit.

**Did you call DeepSeek?**  Not in this acceptance run. A key is configured, but the real evaluation
adapter and real API composition root are not implemented. A provider-only paid call would not prove
end-to-end RAG quality, so it requires separate fee approval and must be reported as such.

**What would you build next?**  Assemble the real request path, add dependency probes and a durable
worker, then run a versioned evaluation that captures provider usage/cost and end-to-end latency.
