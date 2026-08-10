from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai17_markdown_ingestion import DEFAULT_OUTPUT_PATH as DEFAULT_JSONL_PATH
from ai18_jsonl_retrieval import (
    EmbeddedJsonlChunk,
    JsonlChunk,
    RetrievalHit,
    RetrievalMode,
    embed_chunks,
    load_jsonl_chunks,
    retrieve_with_diagnostics,
)


DEFAULT_EVAL_PATH = Path(__file__).resolve().parent / "data" / "bagu_retrieval_eval_v0.jsonl"


@dataclass(frozen=True)
class EvalCase:
    id: str
    split: str
    topic: str
    query_type: str
    query: str
    expected_heading: str | None
    should_find: bool
    notes: str

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> EvalCase:
        return cls(
            id=str(record["id"]),
            split=str(record["split"]),
            topic=str(record["topic"]),
            query_type=str(record["query_type"]),
            query=str(record["query"]),
            expected_heading=optional_str(record.get("expected_heading")),
            should_find=bool(record["should_find"]),
            notes=str(record.get("notes", "")),
        )


@dataclass(frozen=True)
class CaseResult:
    case: EvalCase
    top_headings: list[str]
    expected_rank: int | None
    predicted_gap: bool
    fallback_reason: str | None
    top_score: float
    top_chunk_type: str | None

    @property
    def top1_hit(self) -> bool:
        return self.expected_rank == 1

    def recall_at(self, k: int) -> bool:
        return self.expected_rank is not None and self.expected_rank <= k

    @property
    def reciprocal_rank(self) -> float:
        if self.expected_rank is None:
            return 0.0
        return 1.0 / self.expected_rank

    @property
    def code_noise(self) -> bool:
        return self.case.query_type == "concept" and self.top_chunk_type == "code"


def load_eval_cases(path: Path) -> list[EvalCase]:
    with path.open("r", encoding="utf-8") as file:
        return [
            EvalCase.from_record(json.loads(line))
            for line in file
            if line.strip()
        ]


def evaluate_case(
    case: EvalCase,
    chunks: list[JsonlChunk],
    embedded_chunks: list[EmbeddedJsonlChunk],
    top_k: int,
    mode: RetrievalMode,
) -> CaseResult:
    result = retrieve_with_diagnostics(
        query=case.query,
        chunks=chunks,
        embedded_chunks=embedded_chunks,
        top_k=top_k,
        mode=mode,
    )
    top_headings = [chunk_heading(hit) for hit in result.hits]
    expected_rank = find_expected_rank(case.expected_heading, top_headings)
    top_chunk_type = result.hits[0].chunk.chunk_type if result.hits else None
    return CaseResult(
        case=case,
        top_headings=top_headings,
        expected_rank=expected_rank,
        predicted_gap=result.diagnostics.possible_knowledge_gap,
        fallback_reason=result.diagnostics.fallback_reason,
        top_score=result.diagnostics.top_score,
        top_chunk_type=top_chunk_type,
    )


def chunk_heading(hit: RetrievalHit) -> str:
    if hit.chunk.heading_path:
        return hit.chunk.heading_path[-1]
    return hit.chunk.title


def find_expected_rank(expected_heading: str | None, top_headings: list[str]) -> int | None:
    if expected_heading is None:
        return None
    for index, heading in enumerate(top_headings, start=1):
        if heading == expected_heading:
            return index
    return None


def summarize_results(results: list[CaseResult], split: str, recall_k: int) -> dict[str, float | int]:
    split_results = [result for result in results if result.case.split == split]
    positives = [result for result in split_results if result.case.should_find]
    gaps = [result for result in split_results if not result.case.should_find]

    predicted_gaps = [result for result in split_results if result.predicted_gap]
    true_predicted_gaps = [
        result for result in predicted_gaps if not result.case.should_find
    ]
    false_gaps = [
        result for result in split_results if result.case.should_find and result.predicted_gap
    ]
    code_noise_results = [result for result in positives if result.code_noise]

    return {
        "cases": len(split_results),
        "positive_cases": len(positives),
        "gap_cases": len(gaps),
        "top1_accuracy": ratio(sum(result.top1_hit for result in positives), len(positives)),
        f"recall_at_{recall_k}": ratio(
            sum(result.recall_at(recall_k) for result in positives),
            len(positives),
        ),
        "mrr": ratio(
            sum(result.reciprocal_rank for result in positives),
            len(positives),
        ),
        "gap_precision": ratio(len(true_predicted_gaps), len(predicted_gaps)),
        "gap_recall": ratio(
            sum(result.predicted_gap for result in gaps),
            len(gaps),
        ),
        "false_gap_rate_on_positive": ratio(len(false_gaps), len(positives)),
        "code_noise_rate": ratio(len(code_noise_results), len(positives)),
    }


def ratio(numerator: float, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def print_summary(results: list[CaseResult], recall_k: int) -> None:
    for split in unique_preserve_order(result.case.split for result in results):
        summary = summarize_results(results, split, recall_k)
        print(f"[{split}]")
        for key, value in summary.items():
            print(f"{key}={value}")
        print()


def print_failures(results: list[CaseResult], recall_k: int) -> None:
    failures = [
        result
        for result in results
        if (
            result.case.should_find
            and not result.recall_at(recall_k)
        )
        or (
            not result.case.should_find
            and not result.predicted_gap
        )
        or result.code_noise
    ]
    if not failures:
        print("failures=0")
        return

    print(f"failures={len(failures)}")
    for result in failures:
        expected = result.case.expected_heading or "KNOWLEDGE_GAP"
        observed = " | ".join(result.top_headings[:3]) or "(no hits)"
        print(
            f"- {result.case.id} split={result.case.split} "
            f"expected={expected} rank={result.expected_rank} "
            f"predicted_gap={result.predicted_gap} reason={result.fallback_reason}"
        )
        print(f"  query={result.case.query}")
        print(f"  top3={observed}")


def unique_preserve_order(items: Any) -> list[Any]:
    seen: set[Any] = set()
    result: list[Any] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate AI-18 retrieval on a fixed eval set.")
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL_PATH)
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL_PATH)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--mode", choices=["keyword", "vector", "hybrid"], default="hybrid")
    parser.add_argument("--show-failures", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunks = load_jsonl_chunks(args.jsonl)
    embedded_chunks = embed_chunks(chunks)
    cases = load_eval_cases(args.eval)
    results = [
        evaluate_case(
            case=case,
            chunks=chunks,
            embedded_chunks=embedded_chunks,
            top_k=args.top_k,
            mode=args.mode,
        )
        for case in cases
    ]

    print(f"chunks={len(chunks)}")
    print(f"eval_cases={len(cases)}")
    print(f"mode={args.mode}")
    print(f"top_k={args.top_k}")
    print()
    print_summary(results, args.top_k)
    if args.show_failures:
        print_failures(results, args.top_k)


if __name__ == "__main__":
    main()
