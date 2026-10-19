"""Explicitly non-production services for the offline browser demo."""

# ruff: noqa: RUF001 -- full-width punctuation is intentional Chinese demo content.

from __future__ import annotations

from app.api.dependencies import ApiServices, NoopRequestLimiter
from app.core.config import Settings
from app.core.errors import AppError
from app.core.security import AuthenticationError
from app.schemas.api import Citation, QueryRequest, QueryResponse, Usage
from app.schemas.documents import Principal
from app.services.document_service import InMemoryDocumentService, MemoryIngestQueue

DEMO_BEARER_TOKEN = "demo-local-token"


class DemoAuthenticator:
    """One fixed token, accepted only by the explicitly selected fake runtime."""

    def verify_bearer(self, authorization: str | None) -> Principal:
        if authorization != f"Bearer {DEMO_BEARER_TOKEN}":
            raise AuthenticationError("invalid demo bearer token")
        return Principal(
            user_id="demo-user",
            tenant_id="northstar-demo",
            group_ids=["hr"],
            scopes=["query:read", "documents:read", "documents:write"],
        )


class DemoQueryService:
    """Deterministic UI fixture; it never calls a model, database, or network."""

    async def query(
        self,
        request: QueryRequest,
        principal: Principal,
        *,
        trace_id: str,
    ) -> QueryResponse:
        if "query:read" not in principal.scopes:
            raise AppError(code="forbidden", message="Required scope is missing", status_code=403)
        lowered = request.query.casefold()
        if any(marker in lowered for marker in ("其他租户", "api key", "越权", "密钥")):
            return QueryResponse(
                trace_id=trace_id,
                answer="该请求触及未授权数据或敏感凭据，已被策略拒绝。",
                citations=[],
                usage=Usage(cost_is_estimate=False),
                latency_ms=1,
                retries=0,
                finish_reason="policy_blocked",
            )
        if request.mode == "direct":
            return QueryResponse(
                trace_id=trace_id,
                answer="这是离线 Fake Demo 的直接回答；未访问企业知识库。",
                citations=[],
                usage=Usage(cost_is_estimate=False),
                latency_ms=1,
                retries=0,
                finish_reason="completed",
            )
        retries = 1 if any(marker in lowered for marker in ("重写", "怎么办", "不清楚")) else 0
        return QueryResponse(
            trace_id=trace_id,
            answer=(
                "Northstar 虚构年假制度要求至少提前三个工作日提交申请，"
                "直属经理应在两个工作日内审批。"
            ),
            citations=[
                Citation(
                    document_id="demo-annual-leave",
                    document_version=1,
                    chunk_id="demo-annual-leave-c1",
                    title="年假申请制度",
                    source_uri="demo://northstar/hr/annual-leave",
                    snippet="年假应至少提前三个工作日提交，直属经理在两个工作日内审批。",
                )
            ],
            usage=Usage(cost_is_estimate=False),
            latency_ms=2 + retries,
            retries=retries,
            finish_reason="completed",
        )


def build_demo_services(settings: Settings) -> ApiServices:
    if settings.app_mode != "fake" or settings.app_env != "local":
        raise ValueError("demo services are restricted to local fake mode")
    documents = InMemoryDocumentService(
        MemoryIngestQueue(),
        workspace_memberships={("northstar-demo", "policies", "demo-user")},
        max_bytes=settings.upload_max_bytes,
    )
    return ApiServices(
        authenticator=DemoAuthenticator(),
        documents=documents,
        queries=DemoQueryService(),
        limiter=NoopRequestLimiter(),
        max_upload_bytes=settings.upload_max_bytes,
    )
