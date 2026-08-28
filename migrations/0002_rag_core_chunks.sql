CREATE TABLE document_chunks (
    id TEXT PRIMARY KEY,
    revision_id UUID NOT NULL REFERENCES document_revisions(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('parent', 'child')),
    parent_chunk_id TEXT REFERENCES document_chunks(id),
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    token_count INTEGER NOT NULL CHECK (token_count >= 0),
    tokenizer_name TEXT NOT NULL,
    heading_path JSONB NOT NULL,
    source_uri TEXT NOT NULL,
    line_start INTEGER NOT NULL CHECK (line_start > 0),
    line_end INTEGER NOT NULL CHECK (line_end >= line_start),
    image_references JSONB NOT NULL DEFAULT '[]'::jsonb,
    UNIQUE (revision_id, role, ordinal)
);
CREATE INDEX document_chunks_revision_role_idx ON document_chunks (revision_id, role, ordinal);
