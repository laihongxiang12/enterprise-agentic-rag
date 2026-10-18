# ADR-003: Redis cache, rate limit, and LangGraph checkpoint

- Status: Accepted
- Date: 2026-10-18

## Decision

Use Redis Open Source `8.6.7` and `langgraph-checkpoint-redis==0.5.2` with the project's locked `langgraph==1.2.16`. Redis 8 includes the Search and JSON capabilities required by the official saver. The local container binds only `127.0.0.1:6380`; port 6379 remains untouched because an existing Redis 7 service already uses it.

The official async saver owns checkpoint serialization and persistence. Its Redis Search indices require logical database 0, so both runtime and integration URLs use `/0`. Isolation therefore comes from distinct key/index prefixes plus cryptographic thread IDs, not Redis logical databases.

## Security invariants

- Retrieval cache keys include tenant, workspace, permission hash, allowed index versions, ACL versions, normalized query, top-k, and retrieval configuration hash.
- Only authorized retrieved chunks are cached. Generated answers, prompts, tokens, and credentials are not cached.
- Version or ACL changes produce a cache miss even before explicit workspace-prefix invalidation runs.
- Rate-limit keys derive only from verified tenant, user, and operation values and update atomically with Lua.
- Checkpoint thread IDs hash tenant, user, workspace, session, groups, and scopes. A raw client `session_id` is never used as the Redis thread key.
- Redis calls have bounded connect/read/application timeouts. Query graph concurrency also has a process-local semaphore; multi-hop fan-out retains its existing inner semaphore.

## Consequences

Redis Search index names are global within database 0, so deployments must keep the documented prefixes stable and use a dedicated Redis instance or coordinated prefix migration. The current fixed-window rate limiter is deterministic and atomic but does not claim sliding-window fairness. Redis high availability and production credentials/TLS remain deployment concerns for later stages.

The locked saver currently emits a deprecation warning that its internal async Redis connection factory will become awaitable in a future major release. This is not a runtime failure in `0.5.2`; upgrades must rerun the real recovery test before changing the pin.

## Verification

`tests/unit/test_cache.py` covers key isolation, version/ACL misses, scoped invalidation, Principal-bound checkpoint keys, and query concurrency. `tests/integration/test_checkpoint.py` uses the real Redis container for TTL expiry, concurrent 429 behavior, and checkpoint recovery/isolation.
