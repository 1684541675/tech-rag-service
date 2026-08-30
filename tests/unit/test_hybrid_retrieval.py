import unittest

from rag_core.ingestion import ParentChildMarkdownChunker
from rag_core.retrieval import ParallelHybridRetriever, ParentWindowRetriever, rrf_fuse
from rag_core.stores import KeywordHit, VectorHit


class CharacterTokenizer:
    name = "character-v1"
    def encode(self, text): return [ord(character) for character in text]
    def decode(self, tokens): return "".join(chr(token) for token in tokens)


class HybridRetrievalTest(unittest.TestCase):
    def test_rrf_fuses_rankings_without_using_raw_scores(self):
        sparse = [KeywordHit("child-a", "rev-1", 99.0), KeywordHit("child-b", "rev-1", 1.0)]
        dense = [VectorHit("child-b", "rev-1", 0.01), VectorHit("child-a", "rev-1", 0.99)]
        fused = rrf_fuse({"sparse": sparse, "dense": dense}, revision_id="rev-1", limit=2, rank_constant=10)
        self.assertEqual([hit.chunk_id for hit in fused], ["child-a", "child-b"])
        self.assertEqual(fused[0].source_ranks, (("dense", 2), ("sparse", 1)))

    def test_rrf_rejects_hit_from_other_revision(self):
        with self.assertRaises(ValueError):
            rrf_fuse({"sparse": [KeywordHit("child-a", "old", 1.0)]}, revision_id="new", limit=1)

    def test_parallel_retrieval_degrades_when_one_backend_fails(self):
        retriever = ParallelHybridRetriever(sparse_search=lambda: [KeywordHit("child-a", "rev-1", 1.0)], dense_search=lambda: (_ for _ in ()).throw(TimeoutError()))
        result = retriever.retrieve(revision_id="rev-1", limit=3)
        self.assertEqual([hit.chunk_id for hit in result.hits], ["child-a"])
        self.assertEqual(result.diagnostics, ("dense_unavailable:TimeoutError",))

    def test_parallel_retrieval_fails_only_when_both_backends_fail(self):
        retriever = ParallelHybridRetriever(sparse_search=lambda: (_ for _ in ()).throw(RuntimeError()), dense_search=lambda: (_ for _ in ()).throw(RuntimeError()))
        with self.assertRaises(RuntimeError):
            retriever.retrieve(revision_id="rev-1", limit=3)

    def test_parent_window_deduplicates_parent_and_collects_neighbors(self):
        chunks = ParentChildMarkdownChunker(tokenizer=CharacterTokenizer(), parent_max_tokens=2000, child_max_tokens=30).chunk("# Backend\n\n" + "parent context " * 20, revision_id="rev-1", source_uri="fixture.md")
        hits = rrf_fuse({"sparse": [KeywordHit(chunks.children[1].id, "rev-1", 1.0), KeywordHit(chunks.children[2].id, "rev-1", 1.0)]}, revision_id="rev-1", limit=2)
        windows = ParentWindowRetriever(parents=chunks.parents, children=chunks.children).fetch(hits, neighbor_radius=1)
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].parent.id, chunks.children[1].parent_chunk_id)
        self.assertEqual(windows[0].matched_child_ids, (chunks.children[1].id, chunks.children[2].id))
        self.assertIn(chunks.children[0].id, windows[0].window_child_ids)
        self.assertIn(chunks.children[3].id, windows[0].window_child_ids)


if __name__ == "__main__": unittest.main()
