"""Real OpenSearch verification for the AI-25E revision-isolation contract.

Run after ``docker compose -f docker-compose.opensearch.yml up -d``.
The test owns a uniquely named temporary index and removes it in teardown.
"""
from __future__ import annotations

import os
import unittest
import uuid

from opensearchpy import OpenSearch

from rag_core.stores.opensearch import KeywordRecord, OpenSearchBM25Store


@unittest.skipUnless(os.getenv("RUN_OPENSEARCH_INTEGRATION") == "1", "set RUN_OPENSEARCH_INTEGRATION=1")
class OpenSearchRealIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.client = OpenSearch(hosts=[{"host": "localhost", "port": 9200, "scheme": "http"}])
        self.index_name = f"tech-rag-test-{uuid.uuid4().hex}"
        self.client.indices.create(
            index=self.index_name,
            body={
                "mappings": {
                    "properties": {
                        "revision_id": {"type": "keyword"},
                        "chunk_id": {"type": "keyword"},
                        "content": {"type": "text"},
                        "heading_path": {"type": "keyword"},
                    }
                }
            },
        )
        self.store = OpenSearchBM25Store(client=self.client, index_name=self.index_name)

    def tearDown(self):
        self.client.indices.delete(index=self.index_name, ignore=[404])

    def test_bm25_search_and_delete_are_revision_scoped(self):
        self.store.upsert(
            [
                KeywordRecord("old-point", "revision-old", "child-old", "epoll uses old semantics"),
                KeywordRecord("new-point", "revision-new", "child-new", "epoll supports edge triggered events"),
            ]
        )

        hits = self.store.search(query_text="epoll edge triggered", revision_id="revision-new", limit=3)
        self.assertEqual([hit.chunk_id for hit in hits], ["child-new"])
        self.assertEqual([hit.revision_id for hit in hits], ["revision-new"])

        self.store.delete_revision(revision_id="revision-old")
        count = self.client.count(index=self.index_name, body={"query": {"match_all": {}}})["count"]
        self.assertEqual(count, 1)


if __name__ == "__main__":
    unittest.main()
