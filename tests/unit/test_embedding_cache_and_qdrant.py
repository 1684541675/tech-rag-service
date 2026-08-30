import unittest

from rag_core.embedding import CachingEmbedder, EmbeddingCache
from rag_core.stores import QdrantVectorStore, VectorRecord


class FakeEmbedding:
    name = "fake-v1"
    dimension = 3

    def __init__(self):
        self.calls = 0

    def embed(self, text):
        self.calls += 1
        return (float(len(text)), 1.0, 0.0)


class FakeQdrantClient:
    def __init__(self):
        self.upsert_calls = []
        self.query_calls = []

    def upsert(self, collection_name, points, *, wait):
        self.upsert_calls.append((collection_name, points, wait))

    def query_points(self, collection_name, query, *, query_filter, limit, with_payload):
        self.query_calls.append((collection_name, query, query_filter, limit, with_payload))
        return [{"payload": {"chunk_id": "child-1", "revision_id": "rev-1"}, "score": 0.98}]


class EmbeddingCacheAndQdrantTest(unittest.TestCase):
    def test_same_content_model_and_dimension_hits_cache(self):
        model = FakeEmbedding()
        embedder = CachingEmbedder(model=model, cache=EmbeddingCache())
        first, first_hit = embedder.embed(content_hash="same", text="README chunk")
        second, second_hit = embedder.embed(content_hash="same", text="README chunk")
        self.assertEqual(first, second)
        self.assertFalse(first_hit)
        self.assertTrue(second_hit)
        self.assertEqual(model.calls, 1)

    def test_model_or_dimension_change_does_not_reuse_cache_entry(self):
        cache = EmbeddingCache()
        first_model = FakeEmbedding()
        CachingEmbedder(model=first_model, cache=cache).embed(content_hash="same", text="README chunk")

        class NewModel(FakeEmbedding):
            name = "fake-v2"
            dimension = 2
            def embed(self, text):
                self.calls += 1
                return (2.0, 0.0)

        replacement = NewModel()
        _, cache_hit = CachingEmbedder(model=replacement, cache=cache).embed(content_hash="same", text="README chunk")
        self.assertFalse(cache_hit)
        self.assertEqual(replacement.calls, 1)

    def test_qdrant_write_and_read_are_revision_scoped(self):
        client = FakeQdrantClient()
        store = QdrantVectorStore(client=client, collection_name="rag_chunks")
        store.upsert([VectorRecord("point-1", "rev-1", "child-1", (0.1, 0.2, 0.3))])
        hits = store.search(query_vector=(0.1, 0.2, 0.3), revision_id="rev-1", limit=3)
        self.assertEqual(client.upsert_calls[0][0], "rag_chunks")
        self.assertEqual(client.upsert_calls[0][1][0]["payload"]["revision_id"], "rev-1")
        self.assertEqual(client.query_calls[0][2]["must"][0]["match"]["value"], "rev-1")
        self.assertEqual(hits[0].chunk_id, "child-1")

    def test_qdrant_rejects_non_positive_limit(self):
        store = QdrantVectorStore(client=FakeQdrantClient(), collection_name="rag_chunks")
        with self.assertRaises(ValueError):
            store.search(query_vector=(0.0, 0.0, 0.0), revision_id="rev-1", limit=0)


if __name__ == "__main__":
    unittest.main()
