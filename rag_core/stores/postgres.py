"""PostgreSQL adapter for the AI-25C revision-build protocol."""
from __future__ import annotations
import json
from typing import Any
from rag_core.domain.models import Document, DocumentRevision, RevisionStatus

class PostgresRevisionStore:
    """Uses a caller-provided psycopg-compatible connection."""

    def __init__(self, connection: Any) -> None:
        self.connection = connection

    def _run(self, sql, params=()):
        with self.connection.cursor() as cursor:
            cursor.execute(sql, params)

    def ensure_document(self, doc: Document) -> None:
        self._run("INSERT INTO documents(id,source_uri,source_hash) VALUES(%s,%s,%s) ON CONFLICT(id) DO UPDATE SET source_hash=EXCLUDED.source_hash,updated_at=now()", (doc.id,doc.source_uri,doc.source_hash))
        self.connection.commit()

    def create_revision(self, rev: DocumentRevision) -> None:
        self._run("INSERT INTO document_revisions(id,document_id,content_hash,parser_version,splitter_config_hash,index_status,failure_reason) VALUES(%s,%s,%s,%s,%s,%s,%s)", (rev.id,rev.document_id,rev.content_hash,rev.parser_version,rev.splitter_config_hash,rev.status.value,rev.failure_reason))
        self.connection.commit()

    def save_status(self, rev: DocumentRevision) -> None:
        self._run("UPDATE document_revisions SET index_status=%s,failure_reason=%s WHERE id=%s", (rev.status.value,rev.failure_reason,rev.id))
        self.connection.commit()

    def save_chunks(self, chunks) -> None:
        with self.connection.cursor() as c:
            for x in chunks:
                c.execute("INSERT INTO document_chunks(id,revision_id,role,parent_chunk_id,ordinal,content,content_hash,token_count,tokenizer_name,heading_path,source_uri,line_start,line_end,image_references) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s,%s::jsonb)", (x.id,x.revision_id,x.role.value,x.parent_chunk_id,x.ordinal,x.content,x.content_hash,x.token_count,x.tokenizer_name,json.dumps(x.heading_path),x.source_uri,x.line_start,x.line_end,json.dumps([r.__dict__ for r in x.image_references])))
        self.connection.commit()

    def publish(self, doc: Document, rev: DocumentRevision) -> None:
        if rev.status is not RevisionStatus.READY: raise ValueError("only READY revisions publish")
        try:
            with self.connection.cursor() as c:
                c.execute("SELECT active_revision_id FROM documents WHERE id=%s FOR UPDATE", (doc.id,))
                row = c.fetchone()
                if row is None: raise ValueError("document missing")
                old = row[0]
                c.execute("UPDATE document_revisions SET index_status='ready',published_at=now(),failure_reason=NULL WHERE id=%s AND document_id=%s", (rev.id,doc.id))
                c.execute("UPDATE documents SET active_revision_id=%s,updated_at=now() WHERE id=%s", (rev.id,doc.id))
                if old and old != rev.id: c.execute("UPDATE document_revisions SET index_status='superseded' WHERE id=%s", (old,))
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
