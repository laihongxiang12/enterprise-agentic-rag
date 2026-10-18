"""Trusted identity construction and replaceable service dependencies."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.core.config import Settings
from app.llm.base import LLMProvider
from app.llm.budgets import TokenPrice
from app.llm.fake import FakeLLM
from app.llm.providers import DeepSeekLLM, OpenAICompatibleLLM
from app.retrieval.base import EmbeddingProvider, VectorStore
from app.retrieval.fakes import FakeEmbedding, FakeVectorStore
from app.schemas.documents import DocumentMeta, Principal


@dataclass(frozen=True)
class VerifiedIdentity:
    """Identity claims produced only after token/session verification."""

    user_id: str
    tenant_id: str
    group_ids: tuple[str, ...] = ()
    scopes: tuple[str, ...] = ()


def build_principal(identity: VerifiedIdentity) -> Principal:
    """Map verified internal claims into the domain Principal."""
    return Principal(
        user_id=identity.user_id,
        tenant_id=identity.tenant_id,
        group_ids=list(identity.group_ids),
        scopes=list(identity.scopes),
    )


@runtime_checkable
class PolicyEngine(Protocol):
    def authorize_read(
        self,
        principal: Principal,
        document: DocumentMeta,
        *,
        workspace_authorized: bool,
    ) -> bool: ...

    def authorize_tool(self, principal: Principal, tool_name: str) -> bool: ...


@dataclass(frozen=True)
class ServiceContainer:
    llm: LLMProvider
    embedding: EmbeddingProvider
    vector_store: VectorStore


def build_fake_container() -> ServiceContainer:
    """Build deterministic offline dependencies with no network clients."""
    return ServiceContainer(
        llm=FakeLLM(),
        embedding=FakeEmbedding(),
        vector_store=FakeVectorStore(),
    )


def build_llm_provider(settings: Settings) -> LLMProvider:
    """Keep fake/local offline and construct a real client only in real mode."""
    if settings.app_mode != "real":
        return FakeLLM()
    assert settings.llm_api_key is not None  # enforced by Settings validation
    common = {
        "api_key": settings.llm_api_key.get_secret_value(),
        "base_url": settings.llm_base_url,
        "model": settings.llm_model,
        "price": TokenPrice(
            input_per_million=settings.llm_input_price_per_million,
            output_per_million=settings.llm_output_price_per_million,
        ),
        "timeout_seconds": settings.llm_timeout_seconds,
        "max_attempts": settings.llm_max_attempts,
        "max_concurrency": settings.llm_max_concurrency,
        "max_calls": settings.llm_max_calls,
        "max_total_tokens": settings.llm_max_total_tokens,
        "max_output_tokens": settings.llm_max_output_tokens,
    }
    if settings.llm_provider == "deepseek":
        return DeepSeekLLM(**common)
    return OpenAICompatibleLLM(
        api_key=settings.llm_api_key.get_secret_value(),
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        price=TokenPrice(
            input_per_million=settings.llm_input_price_per_million,
            output_per_million=settings.llm_output_price_per_million,
        ),
        timeout_seconds=settings.llm_timeout_seconds,
        max_attempts=settings.llm_max_attempts,
        max_concurrency=settings.llm_max_concurrency,
        max_calls=settings.llm_max_calls,
        max_total_tokens=settings.llm_max_total_tokens,
        max_output_tokens=settings.llm_max_output_tokens,
        structured_output_mode=settings.llm_structured_output_mode,
    )
