"""Real Qdrant smoke test for the AI-25D revision-scoped vector adapter.

Run explicitly after Docker Desktop and the local Qdrant container are ready:
    python -m unittest tests.integration.test_qdrant_real -v
"""

from __future__ import annotations

import unittest
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from rag_core.stores import QdrantVectorStore, VectorRecord


class RealQdrantVectorStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = QdrantClient(url="http://localhost:6333")
        self.collection_name = f"ai25d_smoke_{uuid4().hex}"
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=3, distance=Distance.COSINE),
        )
        self.store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
        )

    def tearDown(self) -> None:
        self.client.delete_collection(self.collection_name)

    def test_upsert_and_search_only_return_requested_revision(self) -> None:
        self.store.upsert(
            [
                VectorRecord(str(uuid4()), "revision-a", "child-a", (1.0, 0.0, 0.0)),
                VectorRecord(str(uuid4()), "revision-b", "child-b", (0.0, 1.0, 0.0)),
            ]
        )

        hits = self.store.search(
            query_vector=(1.0, 0.0, 0.0),
            revision_id="revision-a",
            limit=3,
        )

        self.assertEqual(
            [(hit.revision_id, hit.chunk_id) for hit in hits],
            [("revision-a", "child-a")],
        )


if __name__ == "__main__":
    unittest.main()
