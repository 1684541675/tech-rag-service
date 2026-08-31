"""Parallel sparse/dense retrieval, rank-only fusion, and parent context lookup."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

from rag_core.ingestion.markdown import IngestedChunk


@dataclass(frozen=True)
class FusedHit:
    chunk_id: str
    revision_id: str
    rrf_score: float
    source_ranks: tuple[tuple[str, int], ...]
    source_scores: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class HybridRetrievalResult:
    hits: tuple[FusedHit, ...]
    diagnostics: tuple[str, ...]


@dataclass(frozen=True)
class ParentWindow:
    parent: IngestedChunk
    matched_child_ids: tuple[str, ...]
    window_child_ids: tuple[str, ...]


def rrf_fuse(ranked_results: Mapping[str, Sequence[object]], *, revision_id: str, limit: int, rank_constant: int = 60) -> tuple[FusedHit, ...]:
    """Fuse rankings only; raw BM25/cosine scores must never be mixed."""
    if not revision_id.strip():
        raise ValueError("revision_id must not be empty")
    if limit <= 0:
        raise ValueError("limit must be positive")
    if rank_constant <= 0:
        raise ValueError("rank_constant must be positive")
    scores: dict[str, float] = {}
    ranks: dict[str, list[tuple[str, int]]] = {}
    source_scores: dict[str, list[tuple[str, float]]] = {}
    for source, hits in ranked_results.items():
        seen: set[str] = set()
        for rank, hit in enumerate(hits, 1):
            if getattr(hit, "revision_id") != revision_id:
                raise ValueError(f"{source} returned a hit from another revision")
            chunk_id = getattr(hit, "chunk_id")
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rank_constant + rank)
            ranks.setdefault(chunk_id, []).append((source, rank))
            source_scores.setdefault(chunk_id, []).append((source, float(getattr(hit, "score"))))
    ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:limit]
    return tuple(
        FusedHit(
            chunk_id,
            revision_id,
            scores[chunk_id],
            tuple(sorted(ranks[chunk_id])),
            tuple(sorted(source_scores[chunk_id])),
        )
        for chunk_id in ordered
    )


class ParallelHybridRetriever:
    """Runs independent sparse/dense retrieval concurrently and preserves failures."""
    def __init__(self, *, sparse_search: Callable[[], Sequence[object]], dense_search: Callable[[], Sequence[object]]) -> None:
        self._searches = {"sparse": sparse_search, "dense": dense_search}

    def retrieve(self, *, revision_id: str, limit: int, rank_constant: int = 60) -> HybridRetrievalResult:
        results: dict[str, Sequence[object]] = {}
        diagnostics: list[str] = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            pending = {executor.submit(search): name for name, search in self._searches.items()}
            for future in as_completed(pending):
                source = pending[future]
                try:
                    results[source] = future.result()
                except Exception as exc:
                    diagnostics.append(f"{source}_unavailable:{type(exc).__name__}")
        if not results:
            raise RuntimeError("all retrieval backends are unavailable")
        return HybridRetrievalResult(rrf_fuse(results, revision_id=revision_id, limit=limit, rank_constant=rank_constant), tuple(sorted(diagnostics)))


class ParentWindowRetriever:
    """Maps child hits to their parent context plus a bounded neighboring window."""
    def __init__(self, *, parents: Iterable[IngestedChunk], children: Iterable[IngestedChunk]) -> None:
        self._parents = {chunk.id: chunk for chunk in parents}
        self._children = {chunk.id: chunk for chunk in children}

    def fetch(self, hits: Sequence[FusedHit], *, neighbor_radius: int = 1, limit: int = 3) -> tuple[ParentWindow, ...]:
        if neighbor_radius < 0:
            raise ValueError("neighbor_radius must not be negative")
        if limit <= 0:
            raise ValueError("limit must be positive")
        matches: dict[str, list[str]] = {}
        order: list[str] = []
        for hit in hits:
            child = self._children.get(hit.chunk_id)
            if child is None or child.parent_chunk_id is None:
                continue
            if child.parent_chunk_id not in matches:
                matches[child.parent_chunk_id] = []
                order.append(child.parent_chunk_id)
            if child.id not in matches[child.parent_chunk_id]:
                matches[child.parent_chunk_id].append(child.id)
        siblings: dict[str, list[IngestedChunk]] = {}
        for child in self._children.values():
            if child.parent_chunk_id:
                siblings.setdefault(child.parent_chunk_id, []).append(child)
        for values in siblings.values():
            values.sort(key=lambda child: child.ordinal)
        windows: list[ParentWindow] = []
        for parent_id in order[:limit]:
            parent = self._parents.get(parent_id)
            if parent is None:
                continue
            values = siblings.get(parent_id, [])
            positions = {child.id: index for index, child in enumerate(values)}
            indexes: set[int] = set()
            for child_id in matches[parent_id]:
                position = positions[child_id]
                indexes.update(range(max(0, position - neighbor_radius), min(len(values), position + neighbor_radius + 1)))
            windows.append(ParentWindow(parent, tuple(matches[parent_id]), tuple(values[index].id for index in sorted(indexes))))
        return tuple(windows)
