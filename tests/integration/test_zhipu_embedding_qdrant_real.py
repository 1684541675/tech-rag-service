"""Explicit paid smoke test: real Zhipu embedding-3 plus local Qdrant."""
from __future__ import annotations

import os
import unittest
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from rag_core.embedding import CachingEmbedder, EmbeddingCache, ZhipuEmbeddingModel
from rag_core.stores import QdrantVectorStore, VectorRecord


@unittest.skipUnless(os.getenv("ZAI_API_KEY"), "ZAI_API_KEY is not configured")
class RealZhipuEmbeddingQdrantTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = ZhipuEmbeddingModel()
        self.client = QdrantClient(url="http://localhost:6333")
        self.collection_name = f"ai25d_zhipu_smoke_{uuid4().hex}"
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.model.dimension, distance=Distance.COSINE),
        )

    def tearDown(self) -> None:
        self.client.delete_collection(self.collection_name)

    def test_real_embedding_is_cached_and_revision_scoped(self) -> None:
        embedder = CachingEmbedder(model=self.model, cache=EmbeddingCache())
        vector, cache_hit = embedder.embed(
            content_hash="epoll-level-edge-smoke-v1",
            text="epoll 的水平触发和边缘触发有什么区别？",
        )
        cached_vector, cached_hit = embedder.embed(
            content_hash="epoll-level-edge-smoke-v1",
            text="epoll 的水平触发和边缘触发有什么区别？",
        )
        self.assertFalse(cache_hit)
        self.assertTrue(cached_hit)
        self.assertEqual(vector, cached_vector)
        self.assertEqual(len(vector), self.model.dimension)

        store = QdrantVectorStore(client=self.client, collection_name=self.collection_name)
        store.upsert(
            [
                VectorRecord(str(uuid4()), "revision-a", "child-a", vector),
                VectorRecord(str(uuid4()), "revision-b", "child-b", vector),
            ]
        )
        hits = store.search(query_vector=vector, revision_id="revision-a", limit=3)
        self.assertEqual([(hit.revision_id, hit.chunk_id) for hit in hits], [("revision-a", "child-a")])


if __name__ == "__main__":
    unittest.main()
