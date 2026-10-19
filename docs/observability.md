# Observability runbook

Stage 18 adds one safe observability path across HTTP, the query service,
LangGraph nodes, retrieval, reranking, LLM calls, and policy checks. Every JSON
operation event and span carries the application `trace_id`; HTTP responses also
return it in `X-Trace-ID` and query responses retain the same value.

## Signals

Structured logs are one JSON object per line. Their vocabulary is allowlisted:
identifiers, outcome, duration, route, bounded counts, and error **type** are
allowed. Query text, prompts, document content, bearer tokens, API keys, user IDs,
and exception messages are not fields and must never be added.

Span names follow `<component>.<operation>`. The important names are:

- `http.request` and `service.query`
- `agent.analyze_query`, `agent.policy_check`, `agent.retrieve`,
  `agent.grade_evidence`, `agent.rewrite_query`, `agent.multihop_plan`,
  `agent.generate`, and `agent.verify`
- terminal agent nodes: `agent.direct_answer`, `agent.insufficient_evidence`, and
  `agent.policy_blocked`
- `retrieval.retrieve`, `rerank.rerank`, `llm.complete`
- `policy.authorize_read` and `policy.authorize_tool`

The local exporter keeps only the newest 1,000 finished spans in memory. It is a
debug/test exporter, not durable production telemetry. A production deployment
must attach its approved OpenTelemetry collector/exporter before relying on trace
retention.

Prometheus exports these metrics:

| Metric | Labels | Meaning |
|---|---|---|
| `enterprise_rag_operations_total` | `component`, `operation`, `outcome` | completed operations |
| `enterprise_rag_operation_duration_seconds` | `component`, `operation` | duration histogram |
| `enterprise_rag_retries_total` | `component` | agent retry count |
| `enterprise_rag_llm_tokens_total` | `kind` | provider-reported prompt/completion/total tokens |
| `enterprise_rag_result_items_total` | `component`, `operation` | bounded retrieval/rerank result counts |

These label sets are fixed and low-cardinality. Never add a user ID, query,
document ID, session ID, or trace ID as a Prometheus label.

## Metrics access

`GET /metrics` is allowed from a loopback client. A non-loopback client must send
a valid bearer token whose trusted identity contains the `metrics:read` scope;
otherwise the endpoint returns `403`. Do not expose this route directly to the
public Internet even when authentication is enabled.

Local check:

```bash
curl --fail http://127.0.0.1:8000/metrics
```

Remote/admin check:

```bash
curl --fail -H "Authorization: Bearer $ADMIN_TOKEN" https://internal-api/metrics
```

## Trace a bad case

1. Copy the `trace_id` from the query response, SSE `trace` event, or
   `X-Trace-ID` response header.
2. Filter structured logs or the trace backend by that exact ID. Do not put the
   ID into a Prometheus label.
3. Inspect `agent.analyze_query` and `route` first. A surprising route points to
   routing/schema fallback rather than retrieval.
4. Inspect `retrieval.retrieve` result count and duration, followed by
   `agent.grade_evidence`. Zero results indicates authorization/index/retrieval;
   nonzero results followed by insufficient evidence indicates grading or source
   coverage.
5. If `retry.count` increased, follow `agent.rewrite_query` and the next retrieval
   and grade spans. The rewritten text is deliberately absent; reproduce it only
   in an authorized local debugger.
6. For generation failures, inspect `llm.complete`, `agent.generate`, and
   `agent.verify`. Compare `error.type`, token counts, durations, and final
   `finish_reason`; never expect prompts or provider exception messages in the
   telemetry.
7. Confirm request usage equals the sum of successful provider calls. Compare it
   with the increase in `enterprise_rag_llm_tokens_total` for operational
   consistency, allowing for other concurrent requests in aggregate metrics.

## Verification

The deterministic test exporter validates span names, trace propagation,
durations, errors, retries, usage totals, metrics authorization, and absence of
sensitive payloads:

```bash
uv run pytest tests/unit/test_observability.py -q
uv run pytest -q
```

This verification is offline and does not call DeepSeek or any other paid API.
Telemetry behavior does not require a real model request.
