"""AI-25C: build a revision before making it query-visible."""
from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from typing import Callable, Protocol

from rag_core.domain.models import Document, DocumentRevision, RevisionStatus
from rag_core.ingestion import ParentChildChunks, stable_content_hash

class Chunker(Protocol):
    parent_max_tokens: int
    child_max_tokens: int
    def chunk(self, markdown: str, *, revision_id: str, source_uri: str) -> ParentChildChunks: ...

class RevisionStore(Protocol):
    def ensure_document(self, document: Document) -> None: ...
    def create_revision(self, revision: DocumentRevision) -> None: ...
    def save_status(self, revision: DocumentRevision) -> None: ...
    def save_chunks(self, chunks: tuple) -> None: ...
    def publish(self, document: Document, revision: DocumentRevision) -> None: ...

@dataclass(frozen=True)
class BuildResult:
    document: Document
    revision: DocumentRevision
    published: bool
    error: str | None = None

def _splitter_hash(chunker: Chunker) -> str:
    tokenizer = getattr(getattr(chunker, "tokenizer", None), "name", "unknown")
    value = "|".join((tokenizer, str(chunker.parent_max_tokens), str(chunker.child_max_tokens)))
    return hashlib.sha256(value.encode()).hexdigest()

class RevisionBuildJob:
    """Persist a complete new revision, validate it, then publish atomically."""

    def __init__(self, *, store: RevisionStore, chunker: Chunker,
                 before_publish: Callable[[ParentChildChunks, DocumentRevision], None] | None = None,
                 id_factory: Callable[[], str] | None = None) -> None:
        self.store = store
        self.chunker = chunker
        self.before_publish = before_publish
        self.id_factory = id_factory or (lambda: str(uuid.uuid4()))

    def build(self, markdown: str, *, source_uri: str) -> BuildResult:
        document = Document(str(uuid.uuid5(uuid.NAMESPACE_URL, source_uri)),
                            source_uri, stable_content_hash(markdown))
        revision = DocumentRevision(self.id_factory(), document.id, document.source_hash,
                                    "markdown-parent-child-v1", _splitter_hash(self.chunker))
        self.store.ensure_document(document)
        self.store.create_revision(revision)
        try:
            revision.transition_to(RevisionStatus.PARSING); self.store.save_status(revision)
            chunks = self.chunker.chunk(markdown, revision_id=revision.id, source_uri=source_uri)
            if not chunks.parents or not chunks.children:
                raise ValueError("revision must contain parent and child chunks")
            revision.transition_to(RevisionStatus.CHUNKED); self.store.save_status(revision)
            revision.transition_to(RevisionStatus.INDEXING); self.store.save_status(revision)
            self.store.save_chunks(chunks.parents + chunks.children)
            revision.transition_to(RevisionStatus.VALIDATING); self.store.save_status(revision)
            self._validate(chunks, revision.id)
            if self.before_publish:
                self.before_publish(chunks, revision)
            revision.transition_to(RevisionStatus.READY)
            self.store.publish(document, revision)
            return BuildResult(document, revision, True)
        except Exception as exc:
            if revision.status is not RevisionStatus.FAILED:
                revision.fail(str(exc)); self.store.save_status(revision)
            return BuildResult(document, revision, False, str(exc))

    @staticmethod
    def _validate(chunks: ParentChildChunks, revision_id: str) -> None:
        parents = {chunk.id for chunk in chunks.parents}
        if any(chunk.revision_id != revision_id for chunk in chunks.parents + chunks.children):
            raise ValueError("chunk revision mismatch")
        if any(chunk.parent_chunk_id not in parents for chunk in chunks.children):
            raise ValueError("child references a missing parent")
