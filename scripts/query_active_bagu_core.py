"""Read the active 八股文 revision and run one real hybrid RAG query."""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import psycopg
from opensearchpy import OpenSearch
from qdrant_client import QdrantClient

from rag_core.embedding import CachingEmbedder, ZhipuEmbeddingModel
from rag_core.generation import RagAnswerService, TokenBudgeter, ZhipuChatClient
from rag_core.ingestion import ChunkRole, IngestedChunk, TiktokenTokenizer, stable_content_hash
from rag_core.retrieval import EvidenceGate, ParallelHybridRetriever, ParentWindowRetriever
from rag_core.stores import OpenSearchBM25Store, PostgresEmbeddingCache, QdrantVectorStore
from scripts.run_real_bagu_core import OPENSEARCH_INDEX, QDRANT_COLLECTION, SOURCE, load_dotenv


def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else "epoll 边缘触发时为什么必须读到 EAGAIN？"
    load_dotenv()
    connection = psycopg.connect(os.environ["RAG_TEST_DATABASE_DSN"])
    with connection.cursor() as cur:
        cur.execute("SELECT active_revision_id FROM documents WHERE source_uri=%s", (str(SOURCE),))
        row = cur.fetchone()
        if row is None or row[0] is None:
            raise RuntimeError("八股文 has no active revision")
        revision_id = str(row[0])
        cur.execute("SELECT id,role,parent_chunk_id,ordinal,content,content_hash,token_count,tokenizer_name,heading_path,source_uri,line_start,line_end FROM document_chunks WHERE revision_id=%s ORDER BY ordinal", (revision_id,))
        chunks = [IngestedChunk(str(r[0]), revision_id, ChunkRole(r[1]), str(r[2]) if r[2] else None, r[3], r[4], r[5], r[6], r[7], tuple(r[8]), r[9], r[10], r[11]) for r in cur.fetchall()]
    parents = [chunk for chunk in chunks if chunk.role is ChunkRole.PARENT]
    children = [chunk for chunk in chunks if chunk.role is ChunkRole.CHILD]
    tokenizer = TiktokenTokenizer()
    embedder = CachingEmbedder(model=ZhipuEmbeddingModel(), cache=PostgresEmbeddingCache(connection))
    keyword = OpenSearchBM25Store(client=OpenSearch(hosts=[{"host":"localhost","port":9200,"scheme":"http"}]), index_name=OPENSEARCH_INDEX)
    vector = QdrantVectorStore(client=QdrantClient(url="http://localhost:6333"), collection_name=QDRANT_COLLECTION)
    query_vector, cache_hit = embedder.embed(content_hash=stable_content_hash(query), text=query)
    result = ParallelHybridRetriever(
        sparse_search=lambda: keyword.search(query_text=query, revision_id=revision_id, limit=6),
        dense_search=lambda: vector.search(query_vector=query_vector, revision_id=revision_id, limit=6),
    ).retrieve(revision_id=revision_id, limit=6)
    windows = ParentWindowRetriever(parents=parents, children=children).fetch(result.hits, limit=3)
    evidence = EvidenceGate().assess(retrieval=result, windows=windows)
    answer = RagAnswerService(budgeter=TokenBudgeter(tokenizer), chat_client=ZhipuChatClient()).answer(query=query, windows=windows, evidence=evidence)
    print(f"revision={revision_id} query_embedding_cache_hit={cache_hit}")
    print(f"fused_hits={[(hit.chunk_id, round(hit.rrf_score, 4), hit.source_ranks) for hit in result.hits]}")
    print(f"windows={[(w.parent.heading_path, w.parent.line_start, w.parent.line_end) for w in windows]}")
    print(f"evidence={evidence.reason} max_dense_score={evidence.max_dense_score}")
    print(f"status={answer.status} degraded={answer.degraded} citations={[(c.heading_path,c.line_start,c.line_end) for c in answer.citations]}")
    print(answer.answer)


if __name__ == "__main__": main()
