"""Qdrant adapter boundary for revision-scoped child-vector storage.

The client is injected so unit tests do not need a Qdrant server or its SDK.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence


@dataclass(frozen=True)
class VectorRecord:
    id: str
    revision_id: str
    chunk_id: str
    vector: tuple[float, ...]


@dataclass(frozen=True)
class VectorHit:
    chunk_id: str
    revision_id: str
    score: float


class QdrantClient(Protocol):
    def upsert(self, collection_name: str, points: Sequence[dict[str, Any]], *, wait: bool) -> Any: ...

    def query_points(
        self,
        collection_name: str,
        query: Sequence[float],
        *,
        query_filter: dict[str, Any],
        limit: int,
        with_payload: bool,
    ) -> Any: ...


class QdrantVectorStore:
    """Writes vectors and always scopes reads to one published revision."""

    def __init__(self, *, client: QdrantClient, collection_name: str) -> None:
        self.client = client
        self.collection_name = collection_name

    def upsert(self, records: Sequence[VectorRecord]) -> None:
        points = [
            {
                "id": record.id,
                "vector": list(record.vector),
                "payload": {"revision_id": record.revision_id, "chunk_id": record.chunk_id},
            }
            for record in records
        ]
        if points:
            self.client.upsert(self.collection_name, points, wait=True)

    def search(self, *, query_vector: Sequence[float], revision_id: str, limit: int) -> tuple[VectorHit, ...]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        result = self.client.query_points(
            self.collection_name,
            list(query_vector),
            query_filter={"must": [{"key": "revision_id", "match": {"value": revision_id}}]},
            limit=limit,
            with_payload=True,
        )
        points = getattr(result, "points", result)
        return tuple(
            VectorHit(
                chunk_id=point["payload"]["chunk_id"] if isinstance(point, dict) else point.payload["chunk_id"],
                revision_id=point["payload"]["revision_id"] if isinstance(point, dict) else point.payload["revision_id"],
                score=point["score"] if isinstance(point, dict) else point.score,
            )
            for point in points
        )
