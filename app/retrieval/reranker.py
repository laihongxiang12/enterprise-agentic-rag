"""Authorized, bounded reranking with deterministic failure fallback."""

from __future__ import annotations

import asyncio
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from app.retrieval.base import Reranker
from app.schemas.retrieval import AuthFilter, RetrievedChunk

_TERM = re.compile(r"[a-z0-9]+|[\u3400-\u4dbf\u4e00-\u9fff]", re.IGNORECASE)


class CrossEncoderModel(Protocol):
    """Minimal synchronous shape implemented by local CrossEncoder models."""

    def predict(self, pairs: Sequence[tuple[str, str]]) -> Sequence[float]: ...


class RerankerResultError(RuntimeError):
    """A reranker returned candidates that cannot be trusted."""


class RerankerAuthorizationError(ValueError):
    """A candidate is outside the trusted authorization scope."""


@dataclass(frozen=True, slots=True)
class RerankOutcome:
    chunks: tuple[RetrievedChunk, ...]
    degraded: bool
    failure_code: str | None = None


class LocalCrossEncoderReranker:
    """Run an injected local synchronous model away from the event loop."""

    def __init__(self, model: CrossEncoderModel) -> None:
        self._model = model

    async def arerank(
        self,
        query: str,
        chunks: Sequence[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        _validate_request(query, top_k)
        if not chunks:
            return []
        pairs = [(query, chunk.content) for chunk in chunks]
        raw_scores = await asyncio.to_thread(self._model.predict, pairs)
        return _rank_from_scores(chunks, raw_scores, top_k)


class FakeReranker:
    """Deterministic offline scorer for contract tests; no model or network."""

    async def arerank(
        self,
        query: str,
        chunks: Sequence[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        _validate_request(query, top_k)
        query_terms = set(_TERM.findall(query.casefold()))
        scores = []
        for chunk in chunks:
            document_terms = set(_TERM.findall(f"{chunk.title} {chunk.content}".casefold()))
            scores.append(float(len(query_terms.intersection(document_terms))))
        return _rank_from_scores(chunks, scores, top_k)


class RerankService:
    """Validate top-20 candidates, rerank to top 5-8, and degrade predictably."""

    def __init__(
        self,
        reranker: Reranker,
        *,
        candidate_limit: int = 20,
        output_top_k: int = 8,
        timeout_seconds: float = 2.0,
    ) -> None:
        if candidate_limit < output_top_k:
            raise ValueError("candidate_limit cannot be smaller than output_top_k")
        if not 5 <= output_top_k <= 8:
            raise ValueError("output_top_k must be between 5 and 8")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._reranker = reranker
        self._candidate_limit = candidate_limit
        self._output_top_k = output_top_k
        self._timeout_seconds = timeout_seconds

    async def arerank(
        self,
        query: str,
        chunks: Sequence[RetrievedChunk],
        auth_filter: AuthFilter,
    ) -> RerankOutcome:
        if not query.strip():
            return RerankOutcome(chunks=(), degraded=False)
        candidates = tuple(chunks[: self._candidate_limit])
        _validate_authorized_candidates(candidates, auth_filter)
        if not candidates:
            return RerankOutcome(chunks=(), degraded=False)
        fallback = candidates[: self._output_top_k]
        try:
            reranked = await asyncio.wait_for(
                self._reranker.arerank(query, candidates, self._output_top_k),
                timeout=self._timeout_seconds,
            )
            _validate_reranker_output(candidates, reranked, self._output_top_k)
        except TimeoutError:
            return RerankOutcome(
                chunks=fallback,
                degraded=True,
                failure_code="timeout",
            )
        except RerankerResultError:
            return RerankOutcome(
                chunks=fallback,
                degraded=True,
                failure_code="invalid_result",
            )
        except Exception:
            return RerankOutcome(
                chunks=fallback,
                degraded=True,
                failure_code="dependency_error",
            )
        return RerankOutcome(chunks=tuple(reranked), degraded=False)


def _validate_request(query: str, top_k: int) -> None:
    if not query.strip():
        raise ValueError("query must not be empty")
    if top_k < 1:
        raise ValueError("top_k must be positive")


def _rank_from_scores(
    chunks: Sequence[RetrievedChunk],
    raw_scores: Sequence[float],
    top_k: int,
) -> list[RetrievedChunk]:
    if len(raw_scores) != len(chunks):
        raise RerankerResultError("reranker returned the wrong score count")
    scored: list[tuple[float, int, RetrievedChunk]] = []
    for rank, (chunk, raw_score) in enumerate(zip(chunks, raw_scores, strict=True)):
        score = float(raw_score)
        if not math.isfinite(score):
            raise RerankerResultError("reranker scores must be finite")
        scored.append((score, rank, chunk))
    scored.sort(key=lambda item: (-item[0], item[1], _identity(item[2])))
    return [
        chunk.model_copy(update={"score": score}, deep=True)
        for score, _rank, chunk in scored[:top_k]
    ]


def _validate_authorized_candidates(
    chunks: Sequence[RetrievedChunk], auth_filter: AuthFilter
) -> None:
    if not (
        auth_filter.allowed_document_ids
        and auth_filter.allowed_index_versions
        and auth_filter.allowed_document_versions
    ):
        if chunks:
            raise RerankerAuthorizationError("authorization metadata is incomplete")
        return
    for chunk in chunks:
        binding = (chunk.document_id, chunk.document_version, chunk.index_version)
        visible = chunk.visibility == "tenant" or bool(
            set(chunk.acl_groups).intersection(auth_filter.group_ids)
        )
        if (
            chunk.tenant_id != auth_filter.tenant_id
            or chunk.workspace_id != auth_filter.workspace_id
            or chunk.document_id not in auth_filter.allowed_document_ids
            or chunk.index_version not in auth_filter.allowed_index_versions
            or binding not in auth_filter.allowed_document_versions
            or not visible
        ):
            raise RerankerAuthorizationError("reranker candidate is outside authorization scope")


def _validate_reranker_output(
    candidates: Sequence[RetrievedChunk],
    reranked: Sequence[RetrievedChunk],
    top_k: int,
) -> None:
    expected_count = min(top_k, len(candidates))
    if len(reranked) != expected_count:
        raise RerankerResultError("reranker returned the wrong chunk count")
    originals = {_identity(chunk): chunk for chunk in candidates}
    seen: set[tuple[str, str, str, int, str]] = set()
    for chunk in reranked:
        identity = _identity(chunk)
        original = originals.get(identity)
        if original is None or identity in seen:
            raise RerankerResultError("reranker injected or duplicated a chunk")
        if original.model_dump(exclude={"score"}) != chunk.model_dump(exclude={"score"}):
            raise RerankerResultError("reranker modified source metadata")
        if chunk.score is None or not math.isfinite(chunk.score):
            raise RerankerResultError("reranker did not provide a finite score")
        seen.add(identity)


def _identity(chunk: RetrievedChunk) -> tuple[str, str, str, int, str]:
    return (
        chunk.tenant_id,
        chunk.workspace_id,
        chunk.document_id,
        chunk.document_version,
        chunk.chunk_id,
    )
