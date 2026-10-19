"""Stage 18 tracing, JSON logging, metrics, and access-control tests."""

from __future__ import annotations

import io
import json

import httpx
import pytest
from fastapi import FastAPI
from pydantic import SecretStr

from app.agents.state import AgentRuntimeContext
from app.api.dependencies import ApiServices, NoopRequestLimiter
from app.core.config import Settings
from app.core.dependencies import VerifiedIdentity
from app.core.security import BasicPolicyEngine, DevTokenService
from app.llm.base import LLMMessage
from app.llm.fake import FakeLLM
from app.main import create_app
from app.observability.instrumentation import ObservedLLM, ObservedPolicyEngine
from app.observability.logging import JsonLogEmitter, build_json_logger
from app.observability.tracing import Observability
from app.retrieval.fakes import FakeRetriever
from app.retrieval.reranker import FakeReranker, RerankService
from app.schemas.api import QueryRequest, QueryResponse, Usage
from app.schemas.documents import Principal
from app.schemas.retrieval import AuthFilter, RetrievedChunk
from app.services.document_service import InMemoryDocumentService, MemoryIngestQueue
from app.services.query_service import QueryService

SECRET = "stage-18-local-secret-that-is-at-least-32-characters"
PRIVATE_QUERY = "PRIVATE_QUERY_PAYROLL_SECRET"
PRIVATE_CONTENT = "PRIVATE_DOCUMENT_SALARY_CONTENT"


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False)


def _route() -> str:
    return _json({"route": "knowledge", "confidence": 1.0, "reason": "internal fact"})


def _grade(*, sufficient: bool) -> str:
    return _json(
        {
            "relevant": True,
            "sufficient": sufficient,
            "confidence": 0.9,
            "missing_aspects": [] if sufficient else ["detail"],
            "recommended_action": "answer" if sufficient else "rewrite",
            "reason": "bounded fixture",
        }
    )


def _answer() -> str:
    return _json(
        {
            "answer": "The authorized policy applies. [1]",
            "citations": [
                {
                    "document_id": "policy",
                    "document_version": 1,
                    "chunk_id": "chunk-1",
                }
            ],
        }
    )


def _chunk(number: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        tenant_id="tenant-a",
        workspace_id="hr",
        document_id="policy",
        document_version=1,
        chunk_id=f"chunk-{number}",
        parent_id=f"parent-{number}",
        content=PRIVATE_CONTENT,
        title="Policy",
        source_uri="fixture:///tenant-a/policy.md",
        visibility="restricted",
        acl_groups=["hr"],
        index_version="index-v1",
    )


def _auth_filter() -> AuthFilter:
    return AuthFilter(
        tenant_id="tenant-a",
        workspace_id="hr",
        group_ids=frozenset({"hr"}),
        allowed_document_ids=frozenset({"policy"}),
        allowed_index_versions=frozenset({"index-v1"}),
        allowed_document_versions=frozenset({("policy", 1, "index-v1")}),
    )


def _observability(stream: io.StringIO | None = None) -> Observability:
    logger = build_json_logger(stream=stream or io.StringIO())
    return Observability(logs=JsonLogEmitter(logger))


@pytest.mark.asyncio
async def test_graph_trace_records_nodes_retries_and_consistent_usage_without_payloads() -> None:
    stream = io.StringIO()
    telemetry = _observability(stream)
    rewritten = "bounded rewritten query"
    per_call = Usage(
        prompt_tokens=2,
        completion_tokens=1,
        total_tokens=3,
        cost_is_estimate=False,
    )
    llm = FakeLLM(
        schema_scripts={
            "RouteDecision": [_route()],
            "EvidenceGrade": [_grade(sufficient=False), _grade(sufficient=True)],
            "QueryRewrite": [_json({"query": rewritten, "reason": "add detail"})],
            "GroundedAnswer": [_answer()],
        },
        usage=per_call,
    )
    runtime = AgentRuntimeContext(
        principal=Principal(user_id="alice", tenant_id="tenant-a", group_ids=["hr"]),
        auth_filter=_auth_filter(),
        llm=llm,
        retriever=FakeRetriever(responses={PRIVATE_QUERY: [_chunk()], rewritten: [_chunk()]}),
        policy_engine=BasicPolicyEngine(),
        now_epoch_ms=lambda: 1_000,
        observability=telemetry,
    )

    result = await QueryService().execute(
        QueryRequest(query=PRIVATE_QUERY), runtime, trace_id="trace-observability"
    )

    assert result["retry_count"] == 1
    assert Usage.model_validate(result["usage"]) == Usage(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        cost_is_estimate=False,
    )
    span_names = [span.name for span in telemetry.finished_spans()]
    for expected in (
        "service.query",
        "agent.analyze_and_route",
        "agent.policy_check",
        "agent.retrieve",
        "retrieval.retrieve",
        "agent.grade_evidence",
        "agent.rewrite_query",
        "agent.generate_grounded",
        "agent.verify_citations",
        "llm.complete",
    ):
        assert expected in span_names
    for span in telemetry.finished_spans():
        assert (span.attributes or {})["app.trace_id"] == "trace-observability"
    metrics = telemetry.metrics.render().decode()
    assert 'enterprise_rag_llm_tokens_total{kind="total"} 15.0' in metrics
    assert 'enterprise_rag_retries_total{component="agent"} 1.0' in metrics
    exported = stream.getvalue() + metrics + repr(telemetry.finished_spans())
    for secret in (PRIVATE_QUERY, PRIVATE_CONTENT, SECRET, "full prompt"):
        assert secret not in exported


@pytest.mark.asyncio
async def test_error_trace_records_exception_type_without_message_or_prompt() -> None:
    stream = io.StringIO()
    telemetry = _observability(stream)
    backend = FakeLLM(scripts={PRIVATE_QUERY: [RuntimeError(f"failed with {SECRET}")]})
    observed = ObservedLLM(backend, telemetry)
    with telemetry.trace_scope("trace-error"), pytest.raises(RuntimeError):
        await observed.complete([LLMMessage(role="user", content=PRIVATE_QUERY)])
    logs = stream.getvalue()
    span = telemetry.finished_spans()[-1]
    assert span.name == "llm.complete"
    assert (span.attributes or {})["error.type"] == "RuntimeError"
    assert "RuntimeError" in logs
    assert PRIVATE_QUERY not in logs
    assert SECRET not in logs


def test_unsafe_log_and_span_fields_are_rejected() -> None:
    telemetry = _observability()
    with (
        telemetry.trace_scope("trace-safe-fields"),
        telemetry.observe(component="test", operation="safe") as observed,
        pytest.raises(ValueError, match="unsafe span attribute"),
    ):
        observed.set_attribute("query", PRIVATE_QUERY)
    with pytest.raises(ValueError, match="unsafe structured log fields"):
        telemetry.logs.emit(
            event_type="unsafe_test",
            trace_id="trace-safe-fields",
            component="test",
            operation="safe",
            outcome="denied",
            fields={"api_key": SECRET},  # type: ignore[dict-item]
        )


@pytest.mark.asyncio
async def test_reranker_and_policy_have_separate_spans() -> None:
    telemetry = _observability()
    chunks = [_chunk(index) for index in range(1, 6)]
    reranker = RerankService(FakeReranker(), output_top_k=5, observability=telemetry)
    policy = ObservedPolicyEngine(BasicPolicyEngine(), telemetry)
    principal = Principal(user_id="alice", tenant_id="tenant-a", scopes=["tools:search_docs"])
    with telemetry.trace_scope("trace-boundaries"):
        result = await reranker.arerank("policy", chunks, _auth_filter())
        assert policy.authorize_tool(principal, "search_docs")
    assert len(result.chunks) == 5
    names = [span.name for span in telemetry.finished_spans()]
    assert "rerank.rerank" in names
    assert "policy.authorize_tool" in names


class ApiQueryService:
    async def query(
        self,
        request: QueryRequest,
        principal: Principal,
        *,
        trace_id: str,
    ) -> QueryResponse:
        del request, principal
        return QueryResponse(
            trace_id=trace_id,
            answer="safe",
            usage=Usage(),
            latency_ms=0,
            retries=0,
            finish_reason="completed",
        )


def _settings() -> Settings:
    return Settings(
        app_env="local",
        app_mode="fake",
        bind_host="127.0.0.1",
        dev_auth_enabled=True,
        dev_auth_secret=SecretStr(SECRET),
    )


def _api(scopes: tuple[str, ...]) -> tuple[FastAPI, str, Observability]:
    configured = _settings()
    authenticator = DevTokenService(configured)
    token = authenticator.issue(
        VerifiedIdentity(
            user_id="operator",
            tenant_id="tenant-a",
            scopes=scopes,
        )
    )
    services = ApiServices(
        authenticator=authenticator,
        documents=InMemoryDocumentService(
            MemoryIngestQueue(),
            workspace_memberships={("tenant-a", "default", "operator")},
            max_bytes=configured.upload_max_bytes,
        ),
        queries=ApiQueryService(),
        limiter=NoopRequestLimiter(),
    )
    telemetry = _observability()
    return create_app(configured, api_services=services, observability=telemetry), token, telemetry


@pytest.mark.asyncio
async def test_http_trace_id_propagates_and_metrics_access_is_restricted() -> None:
    app, query_token, telemetry = _api(("query:read",))
    headers = {"Authorization": f"Bearer {query_token}"}
    remote_transport = httpx.ASGITransport(
        app=app,
        client=("203.0.113.10", 443),  # type: ignore[arg-type]
    )
    local_transport = httpx.ASGITransport(
        app=app,
        client=("127.0.0.1", 12345),  # type: ignore[arg-type]
    )
    async with (
        httpx.AsyncClient(transport=remote_transport, base_url="http://test") as remote,
        httpx.AsyncClient(transport=local_transport, base_url="http://test") as local,
    ):
        query = await remote.post("/v1/query", json={"query": "safe question"}, headers=headers)
        denied = await remote.get("/metrics", headers=headers)
        internal = await local.get("/metrics")
    assert query.status_code == 200
    assert query.headers["x-trace-id"] == query.json()["trace_id"]
    assert denied.status_code == 403
    assert internal.status_code == 200
    assert "enterprise_rag_operations_total" in internal.text
    assert query.headers["x-trace-id"] not in internal.text
    assert telemetry.finished_spans()


@pytest.mark.asyncio
async def test_remote_metrics_requires_explicit_admin_scope() -> None:
    app, token, _ = _api(("metrics:read",))
    transport = httpx.ASGITransport(
        app=app,
        client=("203.0.113.10", 443),  # type: ignore[arg-type]
    )
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/metrics", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
