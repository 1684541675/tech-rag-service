"""Create an AI-25 retrieval eval set tied to the current active revision."""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import psycopg

from scripts.run_real_bagu_core import ROOT, load_dotenv

INPUT = ROOT / "data" / "bagu_retrieval_eval_v0.jsonl"
OUTPUT = ROOT / "data" / "ai25_retrieval_eval_v1.jsonl"


def main() -> None:
    load_dotenv()
    connection = psycopg.connect(os.environ["RAG_TEST_DATABASE_DSN"])
    with connection.cursor() as cur:
        cur.execute("SELECT active_revision_id FROM documents WHERE active_revision_id IS NOT NULL")
        row = cur.fetchone()
        if row is None:
            raise RuntimeError("no active revision")
        revision_id = str(row[0])
        cur.execute("SELECT id,parent_chunk_id,role,heading_path FROM document_chunks WHERE revision_id=%s", (revision_id,))
        rows = cur.fetchall()
    children_by_heading: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for chunk_id, parent_id, role, heading_path in rows:
        if role == "child" and heading_path:
            children_by_heading[heading_path[-1]].append((str(chunk_id), str(parent_id)))
    records, unmatched = [], []
    for line in INPUT.read_text(encoding="utf-8").splitlines():
        old = json.loads(line)
        expected = old.get("expected_heading")
        if old["should_find"]:
            matches = children_by_heading.get(expected, [])
            if not matches:
                unmatched.append(old["id"])
                continue
            child_ids = sorted(chunk_id for chunk_id, _ in matches)
            parent_ids = sorted({parent_id for _, parent_id in matches})
        else:
            child_ids, parent_ids = [], []
        records.append({
            "id": old["id"], "split": old["split"], "topic": old["topic"],
            "query_type": old["query_type"], "query": old["query"],
            "should_find": old["should_find"], "expected_heading": expected,
            "expected_child_ids": child_ids, "expected_parent_ids": parent_ids,
            "revision_id": revision_id, "notes": old.get("notes", ""),
        })
    OUTPUT.write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n", encoding="utf-8")
    print(f"revision={revision_id} cases={len(records)} unmatched={unmatched}")


if __name__ == "__main__":
    main()
