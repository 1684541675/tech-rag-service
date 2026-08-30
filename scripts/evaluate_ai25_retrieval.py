"""Evaluate BM25, dense, and RRF retrieval against the active AI-25 revision."""
from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from pathlib import Path

import psycopg
from opensearchpy import OpenSearch
from qdrant_client import QdrantClient

from rag_core.embedding import CachingEmbedder, ZhipuEmbeddingModel
from rag_core.ingestion import stable_content_hash
from rag_core.retrieval import rrf_fuse
from rag_core.stores import OpenSearchBM25Store, PostgresEmbeddingCache, QdrantVectorStore
from scripts.run_real_bagu_core import OPENSEARCH_INDEX, QDRANT_COLLECTION, load_dotenv

ROOT = Path(__file__).resolve().parents[1]
EVAL_PATH = ROOT / "data" / "ai25_retrieval_eval_v1.jsonl"
RESULT_PATH = ROOT / "data" / "ai25_retrieval_eval_v1_results.jsonl"
TOP_K = 5


def rank_of(ids: list[str], expected: set[str]) -> int | None:
    return next((index for index, chunk_id in enumerate(ids, 1) if chunk_id in expected), None)


def ndcg(ids: list[str], expected: set[str], k: int) -> float:
    dcg = sum(1 / math.log2(index + 1) for index, chunk_id in enumerate(ids[:k], 1) if chunk_id in expected)
    ideal = sum(1 / math.log2(index + 1) for index in range(1, min(len(expected), k) + 1))
    return dcg / ideal if ideal else 0.0


def summarize(rows: list[dict], method: str) -> dict[str, float | int]:
    positives = [row for row in rows if row["should_find"]]
    ranks = [row["ranks"][method] for row in positives]
    return {
        "positive_cases": len(positives),
        "recall_at_1": round(sum(rank == 1 for rank in ranks) / len(positives), 4),
        "recall_at_3": round(sum(rank is not None and rank <= 3 for rank in ranks) / len(positives), 4),
        "recall_at_5": round(sum(rank is not None and rank <= 5 for rank in ranks) / len(positives), 4),
        "mrr": round(sum(1 / rank if rank else 0 for rank in ranks) / len(positives), 4),
        "ndcg_at_3": round(sum(row["ndcg"][method]["3"] for row in positives) / len(positives), 4),
        "ndcg_at_5": round(sum(row["ndcg"][method]["5"] for row in positives) / len(positives), 4),
    }


def main() -> None:
    load_dotenv()
    cases = [json.loads(line) for line in EVAL_PATH.read_text(encoding="utf-8").splitlines() if line]
    connection = psycopg.connect(os.environ["RAG_TEST_DATABASE_DSN"])
    with connection.cursor() as cur:
        cur.execute("SELECT active_revision_id FROM documents WHERE active_revision_id IS NOT NULL")
        revision_id = str(cur.fetchone()[0])
    if any(case["revision_id"] != revision_id for case in cases):
        raise RuntimeError("eval set revision does not match active revision; rebuild the eval set")
    sparse = OpenSearchBM25Store(client=OpenSearch(hosts=[{"host":"localhost","port":9200,"scheme":"http"}]), index_name=OPENSEARCH_INDEX)
    dense = QdrantVectorStore(client=QdrantClient(url="http://localhost:6333"), collection_name=QDRANT_COLLECTION)
    embedder = CachingEmbedder(model=ZhipuEmbeddingModel(), cache=PostgresEmbeddingCache(connection))
    results = []
    for number, case in enumerate(cases, 1):
        vector, cache_hit = embedder.embed(content_hash=stable_content_hash(case["query"]), text=case["query"])
        sparse_hits = sparse.search(query_text=case["query"], revision_id=revision_id, limit=TOP_K)
        dense_hits = dense.search(query_vector=vector, revision_id=revision_id, limit=TOP_K)
        rrf_hits = rrf_fuse({"sparse": sparse_hits, "dense": dense_hits}, revision_id=revision_id, limit=TOP_K)
        expected = set(case["expected_child_ids"])
        ids = {"bm25": [hit.chunk_id for hit in sparse_hits], "dense": [hit.chunk_id for hit in dense_hits], "rrf": [hit.chunk_id for hit in rrf_hits]}
        record = {**case, "query_embedding_cache_hit": cache_hit, "rankings": ids,
                  "ranks": {name: rank_of(values, expected) for name, values in ids.items()},
                  "ndcg": {name: {str(k): round(ndcg(values, expected, k), 4) for k in (3, 5)} for name, values in ids.items()}}
        # Gap heuristic only: no Top-5 candidate appears in both retrievers.
        record["gap_predicted_heuristic"] = not any(len(hit.source_ranks) == 2 for hit in rrf_hits)
        results.append(record)
        print(f"evaluated={number}/{len(cases)} id={case['id']} cache_hit={cache_hit}", flush=True)
    RESULT_PATH.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in results) + "\n", encoding="utf-8")
    for method in ("bm25", "dense", "rrf"):
        print(f"[{method}] {json.dumps(summarize(results, method), ensure_ascii=False)}")
    gaps = [row for row in results if not row["should_find"]]
    print("[gap_heuristic] " + json.dumps({"cases": len(gaps), "recall": round(sum(row['gap_predicted_heuristic'] for row in gaps) / len(gaps), 4)}, ensure_ascii=False))


if __name__ == "__main__": main()
