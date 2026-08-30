"""Build and query a real, revision-scoped RAG Core from 八股文.md.

Requires Docker Desktop with Qdrant/OpenSearch running plus local .env containing
RAG_TEST_DATABASE_DSN and an environment ZAI_API_KEY.  It never prints secrets.
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path

import psycopg
from opensearchpy import OpenSearch
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from rag_core.embedding import CachingEmbedder, ZhipuEmbeddingModel
from rag_core.generation import RagAnswerService, TokenBudgeter, ZhipuChatClient
from rag_core.indexing.revision_build import RevisionBuildJob
from rag_core.ingestion import ParentChildMarkdownChunker, TiktokenTokenizer
from rag_core.retrieval import ParallelHybridRetriever, ParentWindowRetriever
from rag_core.stores import (KeywordRecord, OpenSearchBM25Store, PostgresEmbeddingCache,
                             PostgresRevisionStore, QdrantVectorStore, VectorRecord)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "八股文.md"
QDRANT_COLLECTION = "tech_rag_bagu_vectors"
OPENSEARCH_INDEX = "tech-rag-bagu-chunks"


def load_dotenv() -> None:
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def main() -> None:
    load_dotenv()
    dsn = os.getenv("RAG_TEST_DATABASE_DSN")
    if not dsn:
        raise RuntimeError("RAG_TEST_DATABASE_DSN is required")
    if not os.getenv("ZAI_API_KEY"):
        raise RuntimeError("ZAI_API_KEY is required")
    markdown = SOURCE.read_text(encoding="utf-8")
    connection = psycopg.connect(dsn)
    os_client = OpenSearch(hosts=[{"host": "localhost", "port": 9200, "scheme": "http"}])
    q_client = QdrantClient(url="http://localhost:6333")
    if not os_client.indices.exists(index=OPENSEARCH_INDEX):
        os_client.indices.create(index=OPENSEARCH_INDEX, body={"mappings": {"properties": {"revision_id": {"type": "keyword"}, "chunk_id": {"type": "keyword"}, "content": {"type": "text"}, "heading_path": {"type": "keyword"}}}})
    if not q_client.collection_exists(QDRANT_COLLECTION):
        q_client.create_collection(collection_name=QDRANT_COLLECTION, vectors_config=VectorParams(size=1024, distance=Distance.COSINE))
    keyword = OpenSearchBM25Store(client=os_client, index_name=OPENSEARCH_INDEX)
    vector = QdrantVectorStore(client=q_client, collection_name=QDRANT_COLLECTION)
    embedder = CachingEmbedder(model=ZhipuEmbeddingModel(), cache=PostgresEmbeddingCache(connection))
    chunker = ParentChildMarkdownChunker(tokenizer=TiktokenTokenizer(), parent_max_tokens=900, child_max_tokens=220)
    captured = {}
    def index(chunks, revision):
        print(f"indexing revision={revision.id} parents={len(chunks.parents)} children={len(chunks.children)}")
        keyword.rebuild_revision(revision_id=revision.id, records=[KeywordRecord(c.id, revision.id, c.id, c.content, c.heading_path) for c in chunks.children])
        records = []
        for number, child in enumerate(chunks.children, 1):
            embedding, hit = embedder.embed(content_hash=child.content_hash, text=child.content)
            records.append(VectorRecord(str(uuid.uuid5(uuid.NAMESPACE_URL, child.id)), revision.id, child.id, embedding))
            if number % 50 == 0 or number == len(chunks.children):
                print(f"embedded={number}/{len(chunks.children)}", flush=True)
        vector.upsert(records)
        captured["chunks"] = chunks
    job = RevisionBuildJob(store=PostgresRevisionStore(connection), chunker=chunker, before_publish=index)
    result = job.build(markdown, source_uri=str(SOURCE))
    if not result.published:
        raise RuntimeError(f"build failed: {result.error}")
    chunks = captured["chunks"]
    revision = result.revision.id
    hybrid = ParallelHybridRetriever(
        sparse_search=lambda: keyword.search(query_text="epoll 边缘触发为什么读到 EAGAIN", revision_id=revision, limit=5),
        dense_search=lambda: vector.search(query_vector=embedder.embed(content_hash="query-epoll-eagain-v1", text="epoll 边缘触发为什么读到 EAGAIN")[0], revision_id=revision, limit=5),
    ).retrieve(revision_id=revision, limit=4)
    windows = ParentWindowRetriever(parents=chunks.parents, children=chunks.children).fetch(hybrid.hits, limit=3)
    answer = RagAnswerService(budgeter=TokenBudgeter(chunker.tokenizer), chat_client=ZhipuChatClient()).answer(query="epoll 边缘触发时为什么必须读到 EAGAIN？", windows=windows)
    print(f"published=true revision={revision} fused_hits={len(hybrid.hits)} windows={len(windows)}")
    print(f"status={answer.status} degraded={answer.degraded} citations={len(answer.citations)}")
    print(answer.answer)


if __name__ == "__main__":
    main()
