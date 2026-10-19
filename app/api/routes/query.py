"""Authenticated synchronous and SSE query endpoints."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.dependencies import ApiServices, get_api_services, get_principal, get_trace_id
from app.core.errors import AppError
from app.schemas.api import QueryRequest, QueryResponse
from app.schemas.documents import Principal

router = APIRouter(prefix="/v1/query", tags=["query"])


@router.post("", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    principal: Annotated[Principal, Depends(get_principal)],
    services: Annotated[ApiServices, Depends(get_api_services)],
    trace_id: Annotated[str, Depends(get_trace_id)],
) -> QueryResponse:
    await services.limiter.check(principal, "query")
    return await services.queries.query(request, principal, trace_id=trace_id)


@router.post("/stream")
async def query_stream(
    request: QueryRequest,
    principal: Annotated[Principal, Depends(get_principal)],
    services: Annotated[ApiServices, Depends(get_api_services)],
    trace_id: Annotated[str, Depends(get_trace_id)],
) -> StreamingResponse:
    await services.limiter.check(principal, "query_stream")
    return StreamingResponse(
        _stream_query(services, request, principal, trace_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _stream_query(
    services: ApiServices,
    request: QueryRequest,
    principal: Principal,
    trace_id: str,
) -> AsyncIterator[str]:
    yield _event("trace", {"trace_id": trace_id})
    yield _event(
        "status",
        {
            "state": "running",
            "streaming_mode": "simulated_final_answer_segments",
            "token_level": False,
        },
    )
    try:
        response = await services.queries.query(request, principal, trace_id=trace_id)
        for segment in _segments(response.answer):
            yield _event("delta", {"text": segment})
        yield _event(
            "citations",
            {"citations": [item.model_dump(mode="json") for item in response.citations]},
        )
        yield _event(
            "done",
            {
                "finish_reason": response.finish_reason,
                "usage": response.usage.model_dump(mode="json"),
                "latency_ms": response.latency_ms,
                "retries": response.retries,
            },
        )
    except AppError as exc:
        yield _event("error", {"code": exc.code, "message": exc.message})
    except Exception:
        yield _event(
            "error",
            {"code": "internal_error", "message": "Query processing failed"},
        )


def _segments(answer: str, size: int = 64) -> list[str]:
    return [answer[index : index + size] for index in range(0, len(answer), size)]


def _event(name: str, payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"event: {name}\ndata: {encoded}\n\n"
