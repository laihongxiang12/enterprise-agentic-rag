"""Trusted identity construction and replaceable service dependencies."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.llm.base import LLMProvider
from app.llm.fake import FakeLLM
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
