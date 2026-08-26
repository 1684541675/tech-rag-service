import unittest

from rag_core.domain.errors import InvalidRevisionTransition
from rag_core.domain.models import Document, DocumentRevision, RevisionStatus


def ready_revision(document_id: str = "doc-1") -> DocumentRevision:
    revision = DocumentRevision(
        id="rev-1",
        document_id=document_id,
        content_hash="content-sha256",
        parser_version="markdown-v1",
        splitter_config_hash="splitter-sha256",
    )
    for status in (
        RevisionStatus.PARSING,
        RevisionStatus.CHUNKED,
        RevisionStatus.INDEXING,
        RevisionStatus.VALIDATING,
        RevisionStatus.READY,
    ):
        revision.transition_to(status)
    return revision


class RevisionLifecycleTest(unittest.TestCase):
    def test_happy_path_reaches_ready(self) -> None:
        self.assertEqual(ready_revision().status, RevisionStatus.READY)

    def test_invalid_transition_is_rejected(self) -> None:
        revision = DocumentRevision("rev-1", "doc-1", "content", "markdown-v1", "splitter")
        with self.assertRaises(InvalidRevisionTransition):
            revision.transition_to(RevisionStatus.READY)

    def test_failed_revision_cannot_be_published(self) -> None:
        document = Document("doc-1", "notes.md", "source")
        revision = DocumentRevision("rev-1", "doc-1", "content", "markdown-v1", "splitter")
        revision.fail("parser error")

        with self.assertRaises(ValueError):
            document.publish(revision)
        self.assertIsNone(document.active_revision_id)

    def test_ready_revision_is_the_only_visible_revision(self) -> None:
        document = Document("doc-1", "notes.md", "source")
        document.publish(ready_revision())
        replacement = DocumentRevision("rev-2", "doc-1", "updated", "markdown-v1", "splitter")

        with self.assertRaises(ValueError):
            document.publish(replacement)
        self.assertEqual(document.active_revision_id, "rev-1")

    def test_ready_revision_can_be_superseded_then_deleted(self) -> None:
        revision = ready_revision()
        revision.transition_to(RevisionStatus.SUPERSEDED)
        revision.transition_to(RevisionStatus.DELETING)

        self.assertEqual(revision.status, RevisionStatus.DELETING)


if __name__ == "__main__":
    unittest.main()
