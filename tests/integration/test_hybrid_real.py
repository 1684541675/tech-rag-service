"""Real AI-25F verification using temporary OpenSearch and Qdrant data."""
from __future__ import annotations

import os
import unittest
from uuid import uuid4

from opensearchpy import OpenSearch
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from rag_core.ingestion import ParentChildMarkdownChunker
from rag_core.retrieval import ParallelHybridRetriever, ParentWindowRetriever
from rag_core.stores import KeywordRecord, OpenSearchBM25Store, QdrantVectorStore, VectorRecord


class CharacterTokenizer:
    name = "character-v1"
    def encode(self, text): return [ord(character) for character in text]
    def decode(self, tokens): return "".join(chr(token) for token in tokens)


@unittest.skipUnless(os.getenv("RUN_HYBRID_INTEGRATION") == "1", "set RUN_HYBRID_INTEGRATION=1")
class HybridRealIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.revision_id = "revision-current"
        self.os_client = OpenSearch(hosts=[{"host": "localhost", "port": 9200, "scheme": "http"}])
        self.os_index = f"tech-rag-hybrid-{uuid4().hex}"
        self.os_client.indices.create(index=self.os_index, body={"mappings": {"properties": {"revision_id": {"type": "keyword"}, "chunk_id": {"type": "keyword"}, "content": {"type": "text"}, "heading_path": {"type": "keyword"}}}})
        self.qdrant_client = QdrantClient(url="http://localhost:6333")
        self.collection = f"tech_rag_hybrid_{uuid4().hex}"
        self.qdrant_client.create_collection(collection_name=self.collection, vectors_config=VectorParams(size=3, distance=Distance.COSINE))

    def tearDown(self):
        self.os_client.indices.delete(index=self.os_index, ignore=[404])
        self.qdrant_client.delete_collection(self.collection)

    def test_parallel_rrf_and_parent_window(self):
        chunks = ParentChildMarkdownChunker(tokenizer=CharacterTokenizer(), parent_max_tokens=2000, child_max_tokens=55).chunk("# epoll\n\n" + "edge triggered event loop " * 16, revision_id=self.revision_id, source_uri="fixture.md")
        children = chunks.children[:3]
        sparse = OpenSearchBM25Store(client=self.os_client, index_name=self.os_index)
        sparse.upsert([KeywordRecord(child.id, self.revision_id, child.id, child.content, child.heading_path) for child in children])
        dense = QdrantVectorStore(client=self.qdrant_client, collection_name=self.collection)
        dense.upsert([VectorRecord(str(uuid4()), self.revision_id, child.id, (1.0, 0.0, 0.0)) for child in children])
        hybrid = ParallelHybridRetriever(
            sparse_search=lambda: sparse.search(query_text="edge triggered", revision_id=self.revision_id, limit=3),
            dense_search=lambda: dense.search(query_vector=(1.0, 0.0, 0.0), revision_id=self.revision_id, limit=3),
        )
        result = hybrid.retrieve(revision_id=self.revision_id, limit=3)
        self.assertEqual(result.diagnostics, ())
        self.assertEqual({hit.chunk_id for hit in result.hits}, {child.id for child in children})
        windows = ParentWindowRetriever(parents=chunks.parents, children=chunks.children).fetch(result.hits, neighbor_radius=1)
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].parent.revision_id, self.revision_id)
        self.assertTrue(windows[0].window_child_ids)


if __name__ == "__main__": unittest.main()
