"""Real Redis cache, limiter, and LangGraph checkpoint integration."""

from __future__ import annotations

import asyncio
import operator
from typing import Annotated, TypedDict
from uuid import uuid4

import pytest
from langgraph.graph import END, START, StateGraph
from redis.asyncio import Redis

from app.core.config import Settings
from app.core.errors import AppError
from app.schemas.documents import Principal
from app.stores.checkpoint import checkpoint_config, open_redis_checkpointer
from app.stores.redis import RedisCacheStore, RedisRequestLimiter


class CounterState(TypedDict):
    count: Annotated[int, operator.add]


def redis_url() -> str:
    configured = Settings().test_redis_url
    if configured is None:
        pytest.skip("TEST_REDIS_URL is not configured")
    return configured.get_secret_value()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_redis_cache_ttl_invalidation_and_rate_limit() -> None:
    url = redis_url()
    namespace = uuid4().hex
    client: Redis = Redis.from_url(url, decode_responses=False)
    cache = RedisCacheStore(client)
    try:
        await cache.set(f"test:{namespace}:cache:one", b"value", 1)
        assert await cache.get(f"test:{namespace}:cache:one") == b"value"
        for _ in range(20):
            await asyncio.sleep(0.1)
            if await cache.get(f"test:{namespace}:cache:one") is None:
                break
        assert await cache.get(f"test:{namespace}:cache:one") is None

        await cache.set(f"test:{namespace}:scope:a", b"a", 30)
        await cache.set(f"test:{namespace}:scope:b", b"b", 30)
        assert await cache.invalidate(f"test:{namespace}:scope:") == 2

        limiter = RedisRequestLimiter(
            client,
            requests=2,
            window_seconds=30,
        )
        principal = Principal(user_id=f"alice-{namespace}", tenant_id="tenant-a")

        async def attempt() -> int:
            try:
                await limiter.check(principal, "query")
            except AppError as exc:
                return exc.status_code
            return 200

        results = await asyncio.gather(*(attempt() for _ in range(10)))
        assert results.count(200) == 2
        assert results.count(429) == 8
    finally:
        await client.aclose()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_langgraph_checkpoint_resumes_only_same_principal_scope() -> None:
    url = redis_url()

    def increment(_state: CounterState) -> CounterState:
        return CounterState(count=1)

    builder = StateGraph(CounterState)
    builder.add_node("increment", increment)
    builder.add_edge(START, "increment")
    builder.add_edge("increment", END)

    alice = Principal(
        user_id=f"alice-{uuid4().hex}",
        tenant_id="tenant-a",
        group_ids=["hr"],
        scopes=["query:read"],
    )
    bob = alice.model_copy(update={"user_id": f"bob-{uuid4().hex}"})
    alice_config = checkpoint_config(alice, workspace_id="hr", session_id="session-1")
    bob_config = checkpoint_config(bob, workspace_id="hr", session_id="session-1")

    async with open_redis_checkpointer(url, ttl_minutes=5) as saver:
        graph = builder.compile(checkpointer=saver)
        first = await graph.ainvoke(CounterState(count=1), config=alice_config)
        resumed = await graph.ainvoke(CounterState(count=1), config=alice_config)
        isolated = await graph.ainvoke(CounterState(count=1), config=bob_config)

    assert first["count"] == 2
    assert resumed["count"] == 4
    assert isolated["count"] == 2
