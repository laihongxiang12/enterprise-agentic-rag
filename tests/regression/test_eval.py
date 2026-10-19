"""Stage 19 dataset, metric, ablation, and paid-profile regression tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.metrics import Observation, evaluate
from evals.models import GoldenCase, load_suite
from evals.run_eval import run

DATASETS = Path(__file__).parents[2] / "evals" / "datasets"


def _case(**updates: object) -> GoldenCase:
    values: dict[str, object] = {
        "case_id": "case-1",
        "tenant": "northstar",
        "role": "hr",
        "query": "年假规则",
        "gold_chunk_ids": ["gold"],
        "expected_finish_reason": "completed",
        "category": "hr",
        "expected_route": "knowledge",
        "should_retry": False,
    }
    values.update(updates)
    return GoldenCase.model_validate(values)


def test_committed_dataset_counts_and_annotations_match_authorized_source() -> None:
    suite = load_suite(DATASETS)
    assert len(suite.corpus) == 25
    assert len(suite.golden) == 108
    assert len(suite.injection) == 20
    corpus = {item.chunk_id: item for item in suite.corpus}
    for case in suite.golden:
        for chunk_id in case.gold_chunk_ids:
            assert chunk_id in corpus
            assert corpus[chunk_id].tenant == case.tenant
            assert case.role in corpus[chunk_id].roles
            assert corpus[chunk_id].text.startswith(("Northstar", "Bluepeak"))


def test_metrics_handle_empty_input_without_inventing_measurements() -> None:
    metrics = evaluate((), ())
    assert metrics["sample_size"] == 0
    assert metrics["recall_at_5"] == 0.0
    assert metrics["cost"] is None
    assert metrics["retry_success"] is None


def test_duplicate_rank_does_not_inflate_retrieval_or_citation_metrics() -> None:
    observed = Observation(
        case_id="case-1",
        ranked_chunk_ids=("gold", "gold", "noise"),
        predicted_route="knowledge",
        finish_reason="completed",
        retries=0,
        citation_chunk_ids=("gold", "forged"),
        latency_ms=10,
    )
    metrics = evaluate((_case(),), (observed,))
    assert metrics["recall_at_5"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["ndcg_at_10"] == 1.0
    assert metrics["citation_valid_ratio"] == 0.5


def test_retry_success_requires_an_actual_retry_and_recovered_evidence() -> None:
    case = _case(should_retry=True, canonical_query="年假申请")
    no_retry = Observation(
        case_id="case-1",
        ranked_chunk_ids=("gold",),
        predicted_route="knowledge",
        finish_reason="completed",
        retries=0,
        citation_chunk_ids=("gold",),
        latency_ms=1,
    )
    recovered = Observation(
        case_id="case-1",
        ranked_chunk_ids=("gold",),
        predicted_route="knowledge",
        finish_reason="completed",
        retries=1,
        citation_chunk_ids=("gold",),
        latency_ms=2,
    )
    assert evaluate((case,), (no_retry,))["retry_success"] == 0.0
    assert evaluate((case,), (recovered,))["retry_success"] == 1.0


def test_loader_rejects_gold_outside_tenant_or_role(tmp_path: Path) -> None:
    (tmp_path / "corpus.jsonl").write_text(
        json.dumps(
            {
                "chunk_id": "secret",
                "tenant": "other",
                "workspace": "policies",
                "document_id": "secret-policy",
                "title": "秘密",
                "text": "虚构内容",
                "roles": ["finance"],
                "category": "finance",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "golden.jsonl").write_text(
        _case(gold_chunk_ids=["secret"]).model_dump_json() + "\n", encoding="utf-8"
    )
    (tmp_path / "injection.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="outside case authorization"):
        load_suite(tmp_path)


@pytest.mark.asyncio
async def test_all_four_offline_baselines_are_reproducible_and_labeled() -> None:
    first = await run("small", DATASETS)
    second = await run("small", DATASETS)
    assert first["dataset_counts"] == {"corpus": 25, "golden": 24, "injection": 5}
    first_baselines = first["baselines"]
    second_baselines = second["baselines"]
    assert isinstance(first_baselines, dict)
    assert isinstance(second_baselines, dict)
    assert set(first_baselines) == {"A_dense", "B_hybrid_rrf", "C_rerank", "D_agentic"}
    for name, metrics in first_baselines.items():
        assert isinstance(metrics, dict)
        assert metrics["measurement_kind"] == "deterministic_offline_fixture"
        assert metrics["sample_size"] == 24
        assert metrics["injection_block_rate"] == 1.0
        comparable = dict(metrics)
        comparable.pop("latency_p50_ms")
        comparable.pop("latency_p95_ms")
        expected = dict(second_baselines[name])
        expected.pop("latency_p50_ms")
        expected.pop("latency_p95_ms")
        assert comparable == expected


@pytest.mark.asyncio
async def test_real_profile_never_silently_falls_back_to_fake() -> None:
    with pytest.raises(RuntimeError, match="REAL EVAL NOT RUN"):
        await run("real", DATASETS)
