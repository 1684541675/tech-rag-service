"""PostgreSQL-backed embedding cache for reusable, paid provider vectors."""
from __future__ import annotations

import json
from typing import Any

from rag_core.embedding.cache import CacheKey


class PostgresEmbeddingCache:
    """Persists vectors by content hash, provider model, and vector dimension."""

    def __init__(self, connection: Any) -> None:
        self.connection = connection

    def get(self, key: CacheKey) -> tuple[float, ...] | None:
        with self.connection.cursor() as cursor:
            cursor.execute(
                "SELECT vector FROM embedding_cache "
                "WHERE content_hash=%s AND embedding_model=%s AND embedding_dimension=%s",
                (key.content_hash, key.embedding_model, key.embedding_dimension),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        vector = tuple(float(value) for value in row[0])
        if len(vector) != key.embedding_dimension:
            raise RuntimeError("persisted embedding dimension does not match cache key")
        return vector

    def put(self, key: CacheKey, vector: tuple[float, ...]) -> None:
        if len(vector) != key.embedding_dimension:
            raise ValueError("embedding dimension does not match cache key")
        with self.connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO embedding_cache(content_hash,embedding_model,embedding_dimension,vector) "
                "VALUES(%s,%s,%s,%s::jsonb) "
                "ON CONFLICT(content_hash,embedding_model,embedding_dimension) "
                "DO UPDATE SET vector=EXCLUDED.vector, updated_at=now()",
                (
                    key.content_hash,
                    key.embedding_model,
                    key.embedding_dimension,
                    json.dumps(vector),
                ),
            )
        self.connection.commit()
