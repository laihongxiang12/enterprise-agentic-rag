"""Pure evaluation metrics with stable empty and duplicate-rank behavior."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean

from evals.models import GoldenCase


@dataclass(frozen=True, slots=True)
class Observation:
    case_id: str
    ranked_chunk_ids: tuple[str, ...]
    predicted_route: str
    finish_reason: str
    retries: int
    citation_chunk_ids: tuple[str, ...]
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float | None = None


def evaluate(
    cases: tuple[GoldenCase, ...], observations: tuple[Observation, ...]
) -> dict[str, object]:
    if not cases:
        return _empty_metrics()
    by_id = {item.case_id: item for item in observations}
    if len(by_id) != len(observations):
        raise ValueError("duplicate observation case_id")
    missing = [case.case_id for case in cases if case.case_id not in by_id]
    if missing:
        raise ValueError(f"missing observations: {missing[:3]}")

    retrieval_cases = [case for case in cases if case.gold_chunk_ids]
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    citation_total = 0
    citation_valid = 0
    task_success = 0
    retry_cases = 0
    retry_success = 0
    latencies: list[float] = []
    prompt_tokens = 0
    completion_tokens = 0
    measured_costs: list[float] = []
    route_correct = 0

    for case in cases:
        observed = by_id[case.case_id]
        ranked = _deduplicate(observed.ranked_chunk_ids)
        gold = set(case.gold_chunk_ids)
        if gold:
            recalls.append(len(gold.intersection(ranked[:5])) / len(gold))
            first = next(
                (index for index, item in enumerate(ranked, start=1) if item in gold), None
            )
            reciprocal_ranks.append(0.0 if first is None else 1.0 / first)
            gains = [1.0 if item in gold else 0.0 for item in ranked[:10]]
            dcg = sum(gain / math.log2(index + 1) for index, gain in enumerate(gains, start=1))
            ideal = sum(1.0 / math.log2(index + 1) for index in range(1, min(len(gold), 10) + 1))
            ndcgs.append(dcg / ideal if ideal else 0.0)
        route_correct += observed.predicted_route == case.expected_route
        finish_ok = observed.finish_reason == case.expected_finish_reason
        evidence_ok = not gold or bool(gold.intersection(ranked[:5]))
        citation_ok = not gold or bool(gold.intersection(observed.citation_chunk_ids))
        task_success += finish_ok and evidence_ok and citation_ok
        if case.should_retry:
            retry_cases += 1
            retry_success += observed.retries > 0 and finish_ok and evidence_ok
        citation_total += len(observed.citation_chunk_ids)
        citation_valid += sum(item in ranked for item in observed.citation_chunk_ids)
        latencies.append(max(0.0, observed.latency_ms))
        prompt_tokens += observed.prompt_tokens
        completion_tokens += observed.completion_tokens
        if observed.cost is not None:
            measured_costs.append(observed.cost)

    return {
        "sample_size": len(cases),
        "retrieval_sample_size": len(retrieval_cases),
        "recall_at_5": _average(recalls),
        "mrr": _average(reciprocal_ranks),
        "ndcg_at_10": _average(ndcgs),
        "route_accuracy": route_correct / len(cases),
        "task_success": task_success / len(cases),
        "retry_success": retry_success / retry_cases if retry_cases else None,
        "citation_valid_ratio": citation_valid / citation_total if citation_total else None,
        "latency_p50_ms": _percentile(latencies, 0.50),
        "latency_p95_ms": _percentile(latencies, 0.95),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "cost": sum(measured_costs) if len(measured_costs) == len(cases) else None,
    }


def _empty_metrics() -> dict[str, object]:
    return {
        "sample_size": 0,
        "retrieval_sample_size": 0,
        "recall_at_5": 0.0,
        "mrr": 0.0,
        "ndcg_at_10": 0.0,
        "route_accuracy": 0.0,
        "task_success": 0.0,
        "retry_success": None,
        "citation_valid_ratio": None,
        "latency_p50_ms": 0.0,
        "latency_p95_ms": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost": None,
    }


def _deduplicate(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _average(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
