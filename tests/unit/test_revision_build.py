import unittest
from rag_core.domain.models import RevisionStatus
from rag_core.indexing import RevisionBuildJob
from rag_core.ingestion import ParentChildMarkdownChunker

class Tokenizer:
    name = "character"
    def encode(self, text): return [ord(c) for c in text]
    def decode(self, tokens): return "".join(chr(c) for c in tokens)

class MemoryStore:
    def __init__(self): self.docs, self.revisions, self.chunks = {}, {}, {}
    def ensure_document(self, doc):
        if doc.id in self.docs: doc.active_revision_id = self.docs[doc.id].active_revision_id
        self.docs[doc.id] = doc
    def create_revision(self, rev): self.revisions[rev.id] = rev
    def save_status(self, rev): self.revisions[rev.id] = rev
    def save_chunks(self, chunks):
        for x in chunks: self.chunks[x.id] = x
    def publish(self, doc, rev):
        if doc.active_revision_id: self.revisions[doc.active_revision_id].transition_to(RevisionStatus.SUPERSEDED)
        doc.publish(rev); self.docs[doc.id] = doc

class BrokenChunker:
    parent_max_tokens, child_max_tokens = 10, 5
    def chunk(self, *args, **kwargs): raise RuntimeError("fixture failure")

class RevisionBuildTest(unittest.TestCase):
    def setUp(self):
        self.store = MemoryStore()
        self.chunker = ParentChildMarkdownChunker(tokenizer=Tokenizer(), parent_max_tokens=500, child_max_tokens=80)
        self.ids = iter(("one", "two", "bad"))
    def job(self, chunker=None):
        return RevisionBuildJob(store=self.store, chunker=chunker or self.chunker, id_factory=lambda: next(self.ids))
    def test_failure_keeps_old_active_revision(self):
        first = self.job().build("# x\n\nold", source_uri="fixture.md")
        failed = self.job(BrokenChunker()).build("# x\n\nbroken", source_uri="fixture.md")
        self.assertTrue(first.published); self.assertFalse(failed.published)
        self.assertEqual(self.store.docs[first.document.id].active_revision_id, "one")
        self.assertEqual(failed.revision.status, RevisionStatus.FAILED)
    def test_successful_rebuild_replaces_active_revision(self):
        first = self.job().build("# x\n\nold", source_uri="fixture.md")
        second = self.job().build("# x\n\nnew", source_uri="fixture.md")
        self.assertEqual(second.document.active_revision_id, "two")
        self.assertEqual(self.store.revisions[first.revision.id].status, RevisionStatus.SUPERSEDED)
if __name__ == "__main__": unittest.main()
