"""Principal-scoped LangGraph Redis checkpoint configuration and lifecycle."""

from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.redis.aio import AsyncRedisSaver

from app.schemas.documents import Principal


def checkpoint_config(
    principal: Principal,
    *,
    workspace_id: str,
    session_id: str,
) -> RunnableConfig:
    """Build a non-enumerable thread key bound to identity and permission scope."""
    normalized_workspace = workspace_id.strip()
    normalized_session = session_id.strip()
    if not normalized_workspace or not normalized_session:
        raise ValueError("workspace_id and session_id must not be blank")
    payload = {
        "tenant_id": principal.tenant_id,
        "user_id": principal.user_id,
        "workspace_id": normalized_workspace,
        "session_id": normalized_session,
        "groups": sorted(principal.group_ids),
        "scopes": sorted(principal.scopes),
    }
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    thread_id = f"rag-thread-{hashlib.sha256(encoded.encode()).hexdigest()}"
    return {"configurable": {"thread_id": thread_id, "checkpoint_ns": "agent-v1"}}


@asynccontextmanager
async def open_redis_checkpointer(
    redis_url: str,
    *,
    ttl_minutes: int = 60,
    timeout_seconds: float = 2.0,
) -> AsyncIterator[AsyncRedisSaver]:
    """Open and initialize the version-locked official async Redis saver."""
    if ttl_minutes < 1:
        raise ValueError("ttl_minutes must be positive")
    connection_args = {
        "socket_connect_timeout": timeout_seconds,
        "socket_timeout": timeout_seconds,
    }
    async with AsyncRedisSaver.from_conn_string(
        redis_url,
        connection_args=connection_args,
        ttl={"default_ttl": ttl_minutes, "refresh_on_read": True},
        checkpoint_prefix="rag:checkpoint",
        checkpoint_write_prefix="rag:checkpoint_write",
    ) as saver:
        yield saver
