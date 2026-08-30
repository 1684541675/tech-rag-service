import json
import unittest

from rag_core.embedding import CacheKey
from rag_core.stores import PostgresEmbeddingCache


class FakeCursor:
    def __init__(self, connection):
        self.connection = connection
        self.row = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def execute(self, sql, params):
        if sql.startswith("SELECT"):
            self.row = self.connection.rows.get(tuple(params))
            return
        content_hash, model, dimension, vector = params
        self.connection.rows[(content_hash, model, dimension)] = (json.loads(vector),)

    def fetchone(self):
        return self.row


class FakeConnection:
    def __init__(self):
        self.rows = {}
        self.commits = 0

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.commits += 1


class PostgresEmbeddingCacheTest(unittest.TestCase):
    def test_cache_key_includes_content_model_and_dimension(self):
        connection = FakeConnection()
        cache = PostgresEmbeddingCache(connection)
        first = CacheKey("same-content", "embedding-3", 3)
        changed_model = CacheKey("same-content", "embedding-4", 3)
        changed_dimension = CacheKey("same-content", "embedding-3", 2)

        cache.put(first, (0.1, 0.2, 0.3))

        self.assertEqual(cache.get(first), (0.1, 0.2, 0.3))
        self.assertIsNone(cache.get(changed_model))
        self.assertIsNone(cache.get(changed_dimension))
        self.assertEqual(connection.commits, 1)

    def test_rejects_vector_with_wrong_dimension(self):
        cache = PostgresEmbeddingCache(FakeConnection())
        with self.assertRaises(ValueError):
            cache.put(CacheKey("hash", "embedding-3", 3), (0.1, 0.2))


if __name__ == "__main__":
    unittest.main()
