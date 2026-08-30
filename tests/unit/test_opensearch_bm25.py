import unittest

from rag_core.stores.opensearch import KeywordRecord, OpenSearchBM25Store


class FakeOpenSearchClient:
    def __init__(self):
        self.index_calls = []
        self.search_calls = []
        self.delete_calls = []

    def index(self, **kwargs):
        self.index_calls.append(kwargs)

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        return {
            "hits": {
                "hits": [
                    {
                        "_score": 2.5,
                        "_source": {"chunk_id": "child-current", "revision_id": "rev-current"},
                    }
                ]
            }
        }

    def delete_by_query(self, **kwargs):
        self.delete_calls.append(kwargs)


class OpenSearchBM25StoreTest(unittest.TestCase):
    def setUp(self):
        self.client = FakeOpenSearchClient()
        self.store = OpenSearchBM25Store(client=self.client, index_name="rag-chunks")

    def test_search_uses_native_match_and_active_revision_filter(self):
        hits = self.store.search(query_text="epoll 边缘触发", revision_id="rev-current", limit=3)
        body = self.client.search_calls[0]["body"]
        self.assertEqual(body["query"]["bool"]["must"], [{"match": {"content": {"query": "epoll 边缘触发"}}}])
        self.assertEqual(body["query"]["bool"]["filter"], [{"term": {"revision_id": "rev-current"}}])
        self.assertEqual(hits[0].chunk_id, "child-current")

    def test_rebuild_deletes_only_target_revision_then_indexes_its_records(self):
        record = KeywordRecord("point-1", "rev-next", "child-1", "epoll uses event notifications")
        self.store.rebuild_revision(revision_id="rev-next", records=[record])
        self.assertEqual(
            self.client.delete_calls[0]["body"],
            {"query": {"term": {"revision_id": "rev-next"}}},
        )
        self.assertEqual(self.client.index_calls[0]["body"]["revision_id"], "rev-next")

    def test_rebuild_rejects_mixed_revision_records(self):
        records = [KeywordRecord("point-1", "rev-other", "child-1", "content")]
        with self.assertRaises(ValueError):
            self.store.rebuild_revision(revision_id="rev-next", records=records)
        self.assertEqual(self.client.delete_calls, [])

    def test_search_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            self.store.search(query_text="", revision_id="rev-current", limit=1)
        with self.assertRaises(ValueError):
            self.store.search(query_text="query", revision_id="", limit=1)
        with self.assertRaises(ValueError):
            self.store.search(query_text="query", revision_id="rev-current", limit=0)


if __name__ == "__main__":
    unittest.main()
