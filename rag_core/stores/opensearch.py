"""OpenSearch BM25 adapter for revision-scoped child-chunk retrieval.

The client is injected so the query contract is unit-testable without a
running OpenSearch node.  Queries deliberately use OpenSearch's native text
matching; language-specific scoring rules do not belong in this adapter.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence


@dataclass(frozen=True)
class KeywordRecord:
    """A child chunk indexed for sparse retrieval."""

    id: str
    revision_id: str
    chunk_id: str
    content: str
    heading_path: tuple[str, ...] = ()


@dataclass(frozen=True)
class KeywordHit:
    chunk_id: str
    revision_id: str
    score: float


class OpenSearchClient(Protocol):
    def index(self, *, index: str, id: str, body: dict[str, Any], refresh: bool) -> Any: ...

    def search(self, *, index: str, body: dict[str, Any]) -> Any: ...

    def delete_by_query(
        self, *, index: str, body: dict[str, Any], refresh: bool, conflicts: str
    ) -> Any: ...


class OpenSearchBM25Store:
    """Stores child chunks and prevents a query from crossing revisions."""

    def __init__(self, *, client: OpenSearchClient, index_name: str) -> None:
        self.client = client
        self.index_name = index_name

    def upsert(self, records: Sequence[KeywordRecord]) -> None:
        for record in records:
            self.client.index(
                index=self.index_name,
                id=record.id,
                body={
                    "revision_id": record.revision_id,
                    "chunk_id": record.chunk_id,
                    "content": record.content,
                    "heading_path": list(record.heading_path),
                },
                refresh=True,
            )

    def search(self, *, query_text: str, revision_id: str, limit: int) -> tuple[KeywordHit, ...]:
        if not query_text.strip():
            raise ValueError("query_text must not be empty")
        if not revision_id.strip():
            raise ValueError("revision_id must not be empty")
        if limit <= 0:
            raise ValueError("limit must be positive")

        response = self.client.search(
            index=self.index_name,
            body={
                "size": limit,
                "query": {
                    "bool": {
                        "must": [{"match": {"content": {"query": query_text}}}],
                        "filter": [{"term": {"revision_id": revision_id}}],
                    }
                },
            },
        )
        hits = response["hits"]["hits"] if isinstance(response, dict) else response.hits.hits
        return tuple(
            KeywordHit(
                chunk_id=_source(hit)["chunk_id"],
                revision_id=_source(hit)["revision_id"],
                score=float(_value(hit, "_score", "score")),
            )
            for hit in hits
        )

    def delete_revision(self, *, revision_id: str) -> None:
        if not revision_id.strip():
            raise ValueError("revision_id must not be empty")
        self.client.delete_by_query(
            index=self.index_name,
            body={"query": {"term": {"revision_id": revision_id}}},
            refresh=True,
            conflicts="proceed",
        )

    def rebuild_revision(self, *, revision_id: str, records: Sequence[KeywordRecord]) -> None:
        if any(record.revision_id != revision_id for record in records):
            raise ValueError("all records must belong to the rebuilt revision")
        self.delete_revision(revision_id=revision_id)
        self.upsert(records)


def _source(hit: Any) -> dict[str, Any]:
    return hit["_source"] if isinstance(hit, dict) else hit._source


def _value(hit: Any, dict_name: str, object_name: str) -> Any:
    return hit[dict_name] if isinstance(hit, dict) else getattr(hit, object_name)
