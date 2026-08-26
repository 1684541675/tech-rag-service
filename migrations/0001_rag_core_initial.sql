-- AI-25A: PostgreSQL is the truth source for document visibility and metadata.
-- Search and vector indexes will be derived from these rows in later slices.

CREATE TABLE documents (
    id UUID PRIMARY KEY,
    source_uri TEXT NOT NULL UNIQUE,
    source_hash TEXT NOT NULL,
    active_revision_id UUID NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE document_revisions (
    id UUID PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE RESTRICT,
    content_hash TEXT NOT NULL,
    parser_version TEXT NOT NULL,
    splitter_config_hash TEXT NOT NULL,
    index_status TEXT NOT NULL CHECK (index_status IN (
        'pending', 'parsing', 'chunked', 'indexing', 'validating',
        'ready', 'failed', 'superseded', 'deleting'
    )),
    failure_reason TEXT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    published_at TIMESTAMPTZ NULL,
    UNIQUE (document_id, content_hash, parser_version, splitter_config_hash)
);

ALTER TABLE documents
    ADD CONSTRAINT documents_active_revision_fk
    FOREIGN KEY (active_revision_id) REFERENCES document_revisions(id)
    ON DELETE RESTRICT;

CREATE INDEX document_revisions_document_status_idx
    ON document_revisions (document_id, index_status);
