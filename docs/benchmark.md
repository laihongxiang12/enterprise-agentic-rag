# Stage 19 evaluation and benchmark

## Scope and data version

This benchmark uses only original fictional company policies. It contains 25
authorized source chunks, 108 golden queries (100 knowledge and 8 direct), and
20 prompt-injection/security cases. Every knowledge case references a committed
chunk in `corpus.jsonl`; dataset loading fails on missing, cross-tenant, or
role-inaccessible gold chunks.

Dataset SHA-256:

- `corpus.jsonl`: `cb04723ccfc19f94732609d3f8a90e5b4f502a57b7989a9215ee80b96fe53272`
- `golden.jsonl`: `e6183676486e97920155748986c60577351a018c15c26fc34fb27ae598a2d689`
- `injection.jsonl`: `5b789396e660a64d7df3cb0f3d833fcf85dd3ca39d578f0e36d70131fc3a0ff9`

## Reproduction

Measured on WSL2 Linux 6.6.87.2, 13th Gen Intel Core i7-13700H (20 logical
CPUs), Python 3.12.7, and uv 0.13.0.

```bash
uv run pytest tests/regression/test_eval.py -q
uv run python -m evals.run_eval --profile small
uv run python -m evals.run_eval --profile full --output /tmp/stage19-full.json
```

The `small` profile uses 24 gold and 5 injection cases for CI. The `full`
profile evaluates all committed cases offline. Both are deterministic fixture
measurements: hashing embeddings, in-memory BM25/RRF, lexical-overlap reranking,
and a scripted canonical rewrite for cases explicitly marked `should_retry`.
They do not measure DeepSeek, a real cross-encoder, Milvus latency, HTTP latency,
or answer quality from a production model.

`--profile real` is fail-closed and never falls back to fake results. It requires
the explicit `--allow-paid-api` acknowledgement and an approved real evaluation
adapter. Real DeepSeek results remain `NOT_MEASURED`; having an API key configured
is not permission to spend it automatically.

## Metric definitions

- Recall@5 is the fraction of each case's unique gold chunks found in the first
  five unique results, averaged over knowledge cases.
- MRR uses the first unique relevant rank. nDCG@10 uses binary relevance and an
  ideal list bounded by the number of gold chunks.
- Route accuracy compares the predicted `knowledge`/`direct`/`blocked` route.
- Task success requires the expected finish reason, relevant top-five evidence,
  and a gold citation for knowledge cases.
- Retry success is measured only on `should_retry` cases and requires a retry,
  recovered top-five evidence, and the expected finish reason.
- Citation valid ratio requires each citation ID to occur in that case's
  retrieved ranking. Duplicate ranks are removed before retrieval metrics.
- p50/p95 use per-case in-process wall-clock time. Tokens and cost come only from
  provider usage; offline runs therefore report zero tokens and `NOT_MEASURED`
  cost, not “free production inference.”

## Full offline ablation results

Run date: 2026-10-19. Sample: 108 gold / 20 injection cases.

| Baseline | Recall@5 | MRR | nDCG@10 | Route acc. | Task success | Retry success | Citation valid | p50 ms | p95 ms | Tokens | Cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A Dense-only | 0.7700 | 0.6837 | 0.7056 | 1.0000 | 0.7407 | 0.0000 | 1.0000 | 0.068 | 0.105 | 0 | NOT_MEASURED |
| B Hybrid + RRF | 0.8400 | 0.7425 | 0.7676 | 1.0000 | 0.8148 | 0.0000 | 1.0000 | 0.235 | 0.328 | 0 | NOT_MEASURED |
| C + deterministic rerank | 0.8400 | 0.8133 | 0.8202 | 1.0000 | 0.8426 | 0.0000 | 1.0000 | 0.284 | 0.412 | 0 | NOT_MEASURED |
| D + scripted agentic rewrite | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.288 | 0.524 | 0 | NOT_MEASURED |

All four groups blocked 20/20 inert injection fixtures. This is a deterministic
policy assertion, not a claim that a model-based injection classifier has 100%
real-world accuracy.

## Interpretation and bad cases

- Dense-only missed the top five on 23 of 100 knowledge cases. Stable hashing is
  deliberately small and is not a semantic embedding quality claim.
- Hybrid reduced the misses to 16; BM25 helped exact Chinese policy names and
  numeric/time wording.
- Reranking improved ordering (MRR/nDCG) but not recall because it can only reorder
  candidates already retrieved.
- D is an upper-bound workflow fixture: its canonical rewrite is committed with
  each retry case. Its perfect score proves metric wiring and bounded retry
  behavior, not autonomous DeepSeek quality.
- Sub-millisecond latency measures only this tiny in-memory corpus. Production
  p50/p95 across API, Milvus, reranker, and DeepSeek are `NOT_MEASURED`.

The next honest measurement step is an explicitly approved `real` run with a
versioned embedding model, real reranker, indexed corpus, and DeepSeek usage/cost
captured from provider responses.
