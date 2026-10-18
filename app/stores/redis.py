"""Redis retrieval cache and atomic request limiting."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import cast

from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.core.errors import AppError
from app.schemas.documents import Principal
from app.schemas.retrieval import AuthFilter

_RATE_LIMIT_SCRIPT = """
local current = redis.call('INCR', KEYS[1])
if current == 1 then
  redis.call('PEXPIRE', KEYS[1], ARGV[1])
end
local ttl = redis.call('PTTL', KEYS[1])
return {current, ttl}
"""


@dataclass(frozen=True)
class RetrievalCacheKey:
    """Canonical cache identity containing every authorization/version dimension."""

    tenant_id: str
    workspace_id: str
    permission_hash: str
    index_versions: tuple[str, ...]
    acl_versions: tuple[tuple[str, int], ...]
    query: str
    config_hash: str
    top_k: int

    @classmethod
    def from_request(
        cls,
        auth_filter: AuthFilter,
        query: str,
        *,
        config_hash: str,
        top_k: int,
    ) -> RetrievalCacheKey:
        permission_payload = {
            "groups": sorted(auth_filter.group_ids),
            "documents": sorted(auth_filter.allowed_document_ids),
            "document_versions": sorted(auth_filter.allowed_document_versions),
        }
        permission_hash = _digest(permission_payload)
        return cls(
            tenant_id=auth_filter.tenant_id,
            workspace_id=auth_filter.workspace_id,
            permission_hash=permission_hash,
            index_versions=tuple(sorted(auth_filter.allowed_index_versions)),
            acl_versions=tuple(sorted(auth_filter.allowed_acl_versions)),
            query=" ".join(query.casefold().split()),
            config_hash=config_hash,
            top_k=top_k,
        )

    @property
    def redis_key(self) -> str:
        payload = {
            "tenant_id": self.tenant_id,
            "workspace_id": self.workspace_id,
            "permission_hash": self.permission_hash,
            "index_versions": self.index_versions,
            "acl_versions": self.acl_versions,
            "query": self.query,
            "config_hash": self.config_hash,
            "top_k": self.top_k,
        }
        return f"{self.invalidation_prefix}{_digest(payload)}"

    @property
    def invalidation_prefix(self) -> str:
        scope = _digest({"tenant_id": self.tenant_id, "workspace_id": self.workspace_id})
        return f"rag:retrieval:{scope}:"


class RedisCacheStore:
    """Byte cache with bounded I/O and scan-based scoped invalidation."""

    def __init__(self, client: Redis, *, timeout_seconds: float = 2.0) -> None:
        self._client = client
        self._timeout_seconds = timeout_seconds

    @classmethod
    def from_url(cls, redis_url: str, *, timeout_seconds: float = 2.0) -> RedisCacheStore:
        client = Redis.from_url(
            redis_url,
            decode_responses=False,
            socket_connect_timeout=timeout_seconds,
            socket_timeout=timeout_seconds,
        )
        return cls(client, timeout_seconds=timeout_seconds)

    async def get(self, key: str) -> bytes | None:
        try:
            async with asyncio.timeout(self._timeout_seconds):
                value = await self._client.get(key)
        except (TimeoutError, RedisError) as exc:
            raise AppError(
                code="cache_unavailable",
                message="Retrieval cache is unavailable",
                status_code=503,
            ) from exc
        if value is None:
            return None
        if isinstance(value, bytes):
            return value
        return str(value).encode()

    async def set(self, key: str, value: bytes, ttl_seconds: int) -> None:
        if ttl_seconds < 1:
            raise ValueError("ttl_seconds must be positive")
        try:
            async with asyncio.timeout(self._timeout_seconds):
                await self._client.set(key, value, ex=ttl_seconds)
        except (TimeoutError, RedisError) as exc:
            raise AppError(
                code="cache_unavailable",
                message="Retrieval cache is unavailable",
                status_code=503,
            ) from exc

    async def invalidate(self, key_prefix: str) -> int:
        deleted = 0
        try:
            async with asyncio.timeout(self._timeout_seconds):
                async for key in self._client.scan_iter(match=f"{key_prefix}*", count=100):
                    deleted += int(await self._client.unlink(key))
        except (TimeoutError, RedisError) as exc:
            raise AppError(
                code="cache_unavailable",
                message="Retrieval cache is unavailable",
                status_code=503,
            ) from exc
        return deleted

    async def close(self) -> None:
        await self._client.aclose()


class RedisRequestLimiter:
    """Atomic fixed-window limiter scoped to verified tenant, user, and operation."""

    def __init__(
        self,
        client: Redis,
        *,
        requests: int,
        window_seconds: int,
        timeout_seconds: float = 2.0,
    ) -> None:
        if requests < 1 or window_seconds < 1:
            raise ValueError("rate limit requests and window must be positive")
        self._client = client
        self._requests = requests
        self._window_milliseconds = window_seconds * 1_000
        self._timeout_seconds = timeout_seconds

    @classmethod
    def from_url(
        cls,
        redis_url: str,
        *,
        requests: int,
        window_seconds: int,
        timeout_seconds: float = 2.0,
    ) -> RedisRequestLimiter:
        client = Redis.from_url(
            redis_url,
            decode_responses=False,
            socket_connect_timeout=timeout_seconds,
            socket_timeout=timeout_seconds,
        )
        return cls(
            client,
            requests=requests,
            window_seconds=window_seconds,
            timeout_seconds=timeout_seconds,
        )

    async def check(self, principal: Principal, operation: str) -> None:
        identity = _digest(
            {
                "tenant_id": principal.tenant_id,
                "user_id": principal.user_id,
                "operation": operation,
            }
        )
        key = f"rag:rate:{identity}"
        try:
            async with asyncio.timeout(self._timeout_seconds):
                result = await cast(
                    Awaitable[object],
                    self._client.eval(
                        _RATE_LIMIT_SCRIPT,
                        1,
                        key,
                        str(self._window_milliseconds),
                    ),
                )
        except (TimeoutError, RedisError) as exc:
            raise AppError(
                code="rate_limiter_unavailable",
                message="Request limiter is unavailable",
                status_code=503,
            ) from exc
        if not isinstance(result, list) or not result:
            raise AppError(
                code="rate_limiter_unavailable",
                message="Request limiter returned an invalid result",
                status_code=503,
            )
        if int(result[0]) > self._requests:
            raise AppError(code="rate_limited", message="Too many requests", status_code=429)

    async def close(self) -> None:
        await self._client.aclose()


def _digest(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode()).hexdigest()
