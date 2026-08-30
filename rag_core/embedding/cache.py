"""Embedding cache with a content-and-model identity, not a chunk identity."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class EmbeddingModel(Protocol):
    """Small boundary for a real provider or an offline fake embedding model."""

    name: str
    dimension: int

    def embed(self, text: str) -> tuple[float, ...]: ...


class EmbeddingCacheStore(Protocol):
    """Storage boundary shared by in-memory and persistent embedding caches."""

    def get(self, key: "CacheKey") -> tuple[float, ...] | None: ...

    def put(self, key: "CacheKey", vector: tuple[float, ...]) -> None: ...


@dataclass(frozen=True)
class CacheKey:
    content_hash: str
    embedding_model: str
    embedding_dimension: int


class EmbeddingCache:
    """In-memory implementation; a persistent cache can keep the same key later."""

    def __init__(self) -> None:
        self._values: dict[CacheKey, tuple[float, ...]] = {}

    def get(self, key: CacheKey) -> tuple[float, ...] | None:
        return self._values.get(key)

    def put(self, key: CacheKey, vector: tuple[float, ...]) -> None:
        if len(vector) != key.embedding_dimension:
            raise ValueError("embedding dimension does not match cache key")
        self._values[key] = vector


class CachingEmbedder:
    """Caches provider results by normalized content hash, model name and dimension."""

    def __init__(self, *, model: EmbeddingModel, cache: EmbeddingCacheStore) -> None:
        self.model = model
        self.cache = cache

    def embed(self, *, content_hash: str, text: str) -> tuple[tuple[float, ...], bool]:
        key = CacheKey(content_hash, self.model.name, self.model.dimension)
        cached = self.cache.get(key)
        if cached is not None:
            return cached, True
        vector = tuple(self.model.embed(text))
        self.cache.put(key, vector)
        return vector, False
