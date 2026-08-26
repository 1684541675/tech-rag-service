"""Small, dependency-free domain model for document revisions.

Persistence and service adapters are deliberately excluded from this first
vertical slice. The rules here must hold regardless of which database or
index implementation is added later.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from .errors import InvalidRevisionTransition


class RevisionStatus(StrEnum):
    """Lifecycle states for a revision before and after publication."""

    PENDING = "pending"
    PARSING = "parsing"
    CHUNKED = "chunked"
    INDEXING = "indexing"
    VALIDATING = "validating"
    READY = "ready"
    FAILED = "failed"
    SUPERSEDED = "superseded"
    DELETING = "deleting"


ALLOWED_REVISION_TRANSITIONS: Final[dict[RevisionStatus, frozenset[RevisionStatus]]] = {
    RevisionStatus.PENDING: frozenset({RevisionStatus.PARSING, RevisionStatus.FAILED}),
    RevisionStatus.PARSING: frozenset({RevisionStatus.CHUNKED, RevisionStatus.FAILED}),
    RevisionStatus.CHUNKED: frozenset({RevisionStatus.INDEXING, RevisionStatus.FAILED}),
    RevisionStatus.INDEXING: frozenset({RevisionStatus.VALIDATING, RevisionStatus.FAILED}),
    RevisionStatus.VALIDATING: frozenset({RevisionStatus.READY, RevisionStatus.FAILED}),
    RevisionStatus.READY: frozenset({RevisionStatus.SUPERSEDED}),
    RevisionStatus.FAILED: frozenset(),
    RevisionStatus.SUPERSEDED: frozenset({RevisionStatus.DELETING}),
    RevisionStatus.DELETING: frozenset(),
}


@dataclass
class Document:
    """A stable source document; active_revision_id is the query visibility gate."""

    id: str
    source_uri: str
    source_hash: str
    active_revision_id: str | None = None

    def publish(self, revision: "DocumentRevision") -> None:
        """Make a fully validated revision visible to queries."""
        if revision.document_id != self.id:
            raise ValueError("revision does not belong to this document")
        if revision.status is not RevisionStatus.READY:
            raise ValueError("only a ready revision can be published")
        self.active_revision_id = revision.id


@dataclass
class DocumentRevision:
    """An immutable processing attempt for one version of a document."""

    id: str
    document_id: str
    content_hash: str
    parser_version: str
    splitter_config_hash: str
    status: RevisionStatus = RevisionStatus.PENDING
    failure_reason: str | None = None

    def transition_to(self, target: RevisionStatus) -> None:
        """Advance the lifecycle, recording a failure reason separately when needed."""
        allowed_targets = ALLOWED_REVISION_TRANSITIONS[self.status]
        if target not in allowed_targets:
            raise InvalidRevisionTransition(
                f"cannot transition revision from {self.status} to {target}"
            )
        self.status = target

    def fail(self, reason: str) -> None:
        """Move an in-progress revision to FAILED with an observable reason."""
        if not reason.strip():
            raise ValueError("failure reason must not be empty")
        self.transition_to(RevisionStatus.FAILED)
        self.failure_reason = reason
