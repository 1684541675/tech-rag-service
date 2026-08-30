-- AI-25D: cache paid embeddings across process restarts without using chunk ID.
CREATE TABLE embedding_cache (
    content_hash TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding_dimension INTEGER NOT NULL CHECK (embedding_dimension > 0),
    vector JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (content_hash, embedding_model, embedding_dimension)
);
