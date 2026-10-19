"""Strict contracts and cross-file validation for evaluation datasets."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvalModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class CorpusChunk(EvalModel):
    chunk_id: str = Field(min_length=1)
    tenant: str = Field(min_length=1)
    workspace: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    roles: list[str] = Field(min_length=1)
    category: str = Field(min_length=1)


class GoldenCase(EvalModel):
    case_id: str = Field(min_length=1)
    tenant: str = Field(min_length=1)
    role: str = Field(min_length=1)
    query: str = Field(min_length=1)
    gold_chunk_ids: list[str]
    expected_finish_reason: Literal["completed", "insufficient_evidence", "policy_blocked", "error"]
    category: str = Field(min_length=1)
    expected_route: Literal["knowledge", "direct", "blocked"] = "knowledge"
    canonical_query: str | None = None
    should_retry: bool = False

    @model_validator(mode="after")
    def validate_route_and_gold(self) -> GoldenCase:
        if self.expected_route == "knowledge" and not self.gold_chunk_ids:
            raise ValueError("knowledge cases require gold chunks")
        if self.expected_route != "knowledge" and self.gold_chunk_ids:
            raise ValueError("non-knowledge cases cannot declare gold chunks")
        if self.should_retry and not self.canonical_query:
            raise ValueError("retry cases require canonical_query")
        return self


class InjectionCase(EvalModel):
    case_id: str = Field(min_length=1)
    tenant: str = Field(min_length=1)
    role: str = Field(min_length=1)
    query: str = Field(min_length=1)
    gold_chunk_ids: list[str] = Field(default_factory=list, max_length=0)
    expected_finish_reason: Literal["policy_blocked"] = "policy_blocked"
    category: Literal["injection"] = "injection"


class EvalSuite(BaseModel):
    corpus: tuple[CorpusChunk, ...]
    golden: tuple[GoldenCase, ...]
    injection: tuple[InjectionCase, ...]


def load_suite(dataset_dir: Path) -> EvalSuite:
    corpus = tuple(
        CorpusChunk.model_validate(item) for item in _read_jsonl(dataset_dir / "corpus.jsonl")
    )
    golden = tuple(
        GoldenCase.model_validate(item) for item in _read_jsonl(dataset_dir / "golden.jsonl")
    )
    injection = tuple(
        InjectionCase.model_validate(item) for item in _read_jsonl(dataset_dir / "injection.jsonl")
    )
    _unique((item.chunk_id for item in corpus), "chunk_id")
    _unique(
        [item.case_id for item in golden] + [item.case_id for item in injection],
        "case_id",
    )
    chunks = {item.chunk_id: item for item in corpus}
    for case in golden:
        for chunk_id in case.gold_chunk_ids:
            chunk = chunks.get(chunk_id)
            if chunk is None:
                raise ValueError(f"unknown gold chunk: {chunk_id}")
            if chunk.tenant != case.tenant or case.role not in chunk.roles:
                raise ValueError(f"gold chunk is outside case authorization: {case.case_id}")
    return EvalSuite(corpus=corpus, golden=golden, injection=injection)


def _read_jsonl(path: Path) -> list[object]:
    records: list[object] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path.name}:{line_number}") from exc
    return records


def _unique(values: Iterable[str], label: str) -> None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise ValueError(f"duplicate {label}: {value}")
        seen.add(value)
