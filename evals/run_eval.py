"""CLI for deterministic small/full evaluation and guarded real evaluation."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from evals.baselines import BaselineName, OfflineBaselines
from evals.metrics import evaluate
from evals.models import load_suite

BASELINES: tuple[BaselineName, ...] = ("A_dense", "B_hybrid_rrf", "C_rerank", "D_agentic")


async def run(profile: str, dataset_dir: Path) -> dict[str, object]:
    suite = load_suite(dataset_dir)
    if profile == "real":
        raise RuntimeError(
            "REAL EVAL NOT RUN: this profile may call paid APIs; provide an approved real "
            "adapter and explicit --allow-paid-api before measuring"
        )
    cases = suite.golden[:24] if profile == "small" else suite.golden
    injections = suite.injection[:5] if profile == "small" else suite.injection
    runner = await OfflineBaselines.build(suite.corpus)
    results: dict[str, object] = {
        "profile": profile,
        "dataset_counts": {
            "corpus": len(suite.corpus),
            "golden": len(cases),
            "injection": len(injections),
        },
        "baselines": {},
    }
    baselines = results["baselines"]
    assert isinstance(baselines, dict)
    for baseline in BASELINES:
        observations = []
        for case in cases:
            observations.append(await runner.run_case(baseline, case))
        metrics = evaluate(cases, tuple(observations))
        metrics["injection_block_rate"] = (
            sum(runner.run_injection(case) for case in injections) / len(injections)
            if injections
            else None
        )
        metrics["measurement_kind"] = "deterministic_offline_fixture"
        baselines[baseline] = metrics
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("small", "full", "real"), default="small")
    parser.add_argument("--dataset-dir", type=Path, default=Path(__file__).parent / "datasets")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-paid-api", action="store_true")
    args = parser.parse_args()
    if args.profile == "real" and not args.allow_paid_api:
        parser.error("--profile real may incur API cost; pass --allow-paid-api explicitly")
    report = asyncio.run(run(args.profile, args.dataset_dir))
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
