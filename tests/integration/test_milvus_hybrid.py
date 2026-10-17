"""Real Milvus native BM25 + dense + RRF authorization integration."""

import json
from collections.abc import Sequence
from uuid import uuid4

import pytest

from app.core.config import Settings
from app.retrieval.hybrid import HybridRetriever
from app.schemas.retrieval import AuthFilter, RetrievedChunk
from app.stores.milvus import MilvusVectorStore

pytestmark = pytest.mark.integration


class _QueryEmbedding:
    def __init__(self, vector: Sequence[float]) -> None:
        self._vector = list(vector)

    async def embed_query(self, text: str) -> list[float]:
        del text
        return list(self._vector)

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [list(self._vector) for _ in texts]


def _chunk(
    chunk_id: str,
    content: str,
    *,
    tenant_id: str,
    document_id: str,
    group: str = "hr",
    document_version: int = 1,
) -> RetrievedChunk:
    return RetrievedChunk(
        tenant_id=tenant_id,
        workspace_id="hr",
        document_id=document_id,
        document_version=document_version,
        chunk_id=chunk_id,
        parent_id=f"parent-{chunk_id}",
        content=content,
        title="员工制度",
        source_uri=f"fixture:///{tenant_id}/{document_id}",
        visibility="restricted",
        acl_groups=[group],
        heading_path=["人力资源"],
    )


@pytest.mark.asyncio
async def test_real_milvus_chinese_bm25_dense_rrf_and_authorization() -> None:
    uri = Settings().test_milvus_uri
    if uri is None:
        pytest.skip("TEST_MILVUS_URI is not configured for real Milvus integration")

    suffix = uuid4().hex[:12]
    collection = f"stage08_{suffix}"
    tenant_a = f"tenant-a-{suffix}"
    tenant_b = f"tenant-b-{suffix}"
    active_index = f"active-{suffix}"
    stale_index = f"stale-{suffix}"
    exact = _chunk(
        "exact",
        "员工年假申请必须提前三个工作日提交。",
        tenant_id=tenant_a,
        document_id="leave-exact",
    )
    synonym = _chunk(
        "synonym",
        "带薪休假权益按照员工入职年限计算。",
        tenant_id=tenant_a,
        document_id="leave-semantic",
    )
    finance = _chunk(
        "finance-only",
        "年假申请涉及薪资结算机密。",
        tenant_id=tenant_a,
        document_id="finance",
        group="finance",
    )
    other_tenant = _chunk(
        "other-tenant",
        "员工年假申请无需审批。",
        tenant_id=tenant_b,
        document_id="leave-exact",
    )
    stale = _chunk(
        "stale",
        "旧版员工年假申请当天即可提交。",
        tenant_id=tenant_a,
        document_id="leave-exact",
    )
    store = MilvusVectorStore(uri, collection, dimension=4)
    try:
        await store.upsert(
            [exact, synonym, finance, other_tenant],
            [
                [0.1, 0.9, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
            active_index,
        )
        await store.upsert([stale], [[1.0, 0.0, 0.0, 0.0]], stale_index)
        scope = AuthFilter(
            tenant_id=tenant_a,
            workspace_id="hr",
            group_ids=frozenset({"hr"}),
            allowed_document_ids=frozenset({"leave-exact", "leave-semantic", "finance"}),
            allowed_index_versions=frozenset({active_index}),
            allowed_document_versions=frozenset(
                {
                    ("leave-exact", 1, active_index),
                    ("leave-semantic", 1, active_index),
                    ("finance", 1, active_index),
                }
            ),
        )

        lexical = await store.search_lexical("年假申请", scope, top_k=30)
        assert lexical
        assert lexical[0].chunk_id == "exact"
        assert {item.chunk_id for item in lexical}.isdisjoint(
            {"finance-only", "other-tenant", "stale"}
        )

        dense = await store.search_dense([1.0, 0.0, 0.0, 0.0], scope, top_k=30)
        assert dense[0].chunk_id == "synonym"
        assert {item.chunk_id for item in dense}.isdisjoint(
            {"finance-only", "other-tenant", "stale"}
        )

        retriever = HybridRetriever(store, _QueryEmbedding([1.0, 0.0, 0.0, 0.0]))
        run = await retriever.aretrieve_with_baseline("年假申请", scope)
        fused_ids = [item.chunk_id for item in run.fused]
        assert "exact" in fused_ids
        assert "synonym" in fused_ids
        assert set(fused_ids).isdisjoint({"finance-only", "other-tenant", "stale"})
        exported = run.to_eval_record()
        assert exported["parameters"] == {
            "dense_top_k": 30,
            "lexical_top_k": 30,
            "rrf_k": 60,
            "fusion_top_k": 20,
        }
        json.dumps(exported, ensure_ascii=False)

        assert await store.search_lexical("火星量子咖啡条例", scope, top_k=30) == []
        assert await retriever.aretrieve("", scope) == []

        no_groups = scope.model_copy(update={"group_ids": frozenset()})
        assert await store.search_lexical("年假申请", no_groups, top_k=30) == []

        other_scope = scope.model_copy(
            update={
                "tenant_id": tenant_b,
                "allowed_document_ids": frozenset({"leave-exact"}),
                "allowed_document_versions": frozenset({("leave-exact", 1, active_index)}),
            }
        )
        assert [
            item.chunk_id for item in await store.search_lexical("年假申请", other_scope, top_k=30)
        ] == ["other-tenant"]
    finally:
        await store.drop_collection()
        await store.close()
