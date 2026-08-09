from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Any, Iterable, Literal

from ai17_markdown_ingestion import DEFAULT_OUTPUT_PATH as DEFAULT_JSONL_PATH


RetrievalMode = Literal["keyword", "vector", "hybrid"]

TOKEN_RE = re.compile(r"[a-zA-Z0-9_+#.]+|[\u4e00-\u9fff]+")
HEADING_NUMBER_RE = re.compile(r"^[一二三四五六七八九十百零〇两]+\s*、\s*")

EN_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "why",
    "with",
}

QUESTION_STOP_PHRASES = (
    "分别是什么",
    "有什么区别",
    "常见情况",
    "为什么",
    "有哪些",
    "是什么",
    "怎么办",
    "什么",
    "如何",
    "怎么",
    "哪些",
    "是否",
    "可以",
    "进行",
    "使用",
    "实现",
    "解决",
    "区别",
    "分别",
    "常见",
    "情况",
    "关系",
    "不是",
)

CJK_STOP_WORDS = {
    "一个",
    "这个",
    "那个",
    "以及",
    "或者",
    "并且",
    "如果",
    "因为",
    "所以",
    "什么",
    "如何",
    "怎么",
    "哪些",
    "是否",
    "可以",
    "进行",
    "使用",
    "实现",
    "解决",
    "区别",
    "为什么",
    "分别",
    "常见",
    "情况",
    "关系",
    "的是",
    "有什么",
    "有哪些",
    "是什么",
    "怎么办",
    "和",
    "与",
    "的",
    "了",
    "在",
}

DOMAIN_PHRASES = (
    "缓存穿透",
    "缓存击穿",
    "缓存雪崩",
    "水平触发",
    "边缘触发",
    "三次握手",
    "两次握手",
    "四次挥手",
    "虚函数表",
    "虚函数",
    "纯虚函数",
    "索引失效",
    "最左前缀",
    "覆盖索引",
    "倒排索引",
    "线程池",
    "红黑树",
    "多态",
    "epoll",
)


@dataclass(frozen=True)
class JsonlChunk:
    id: str
    source: str
    title: str
    heading_path: list[str]
    chunk_type: str
    text: str
    tables: list[dict[str, Any]]
    code_blocks: list[dict[str, Any]]
    image_urls: list[str]
    metadata: dict[str, Any]

    @property
    def line_start(self) -> int | None:
        return optional_int(self.metadata.get("line_start"))

    @property
    def line_end(self) -> int | None:
        return optional_int(self.metadata.get("line_end"))

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> JsonlChunk:
        return cls(
            id=str(record.get("id", "")),
            source=str(record.get("source", "")),
            title=str(record.get("title", "")),
            heading_path=[str(item) for item in record.get("heading_path", [])],
            chunk_type=str(record.get("chunk_type", "text")),
            text=str(record.get("text", "")),
            tables=list(record.get("tables", [])),
            code_blocks=list(record.get("code_blocks", [])),
            image_urls=[str(item) for item in record.get("image_urls", [])],
            metadata=dict(record.get("metadata", {})),
        )


@dataclass(frozen=True)
class EmbeddedJsonlChunk:
    chunk: JsonlChunk
    vector: list[float]


@dataclass(frozen=True)
class RetrievalHit:
    chunk: JsonlChunk
    score: float
    mode: RetrievalMode


def load_jsonl_chunks(path: Path = DEFAULT_JSONL_PATH) -> list[JsonlChunk]:
    chunks: list[JsonlChunk] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            chunks.append(JsonlChunk.from_record(json.loads(stripped)))
    return chunks


def build_search_text(chunk: JsonlChunk) -> str:
    parts = [
        build_title_search_text(chunk),
        chunk.text,
    ]

    for table in chunk.tables:
        parts.append(str(table.get("markdown", "")))
        for row in table.get("rows", []):
            parts.append(" ".join(f"{key}={value}" for key, value in row.items()))

    for code_block in chunk.code_blocks:
        parts.append(str(code_block.get("language", "")))
        parts.append(str(code_block.get("content", "")))

    return "\n".join(part for part in parts if part)


def build_title_search_text(chunk: JsonlChunk) -> str:
    title_parts = [chunk.title]
    title_parts.extend(clean_heading_title(title) for title in chunk.heading_path)
    return "\n".join(unique_preserve_order(part for part in title_parts if part))


def clean_heading_title(title: str) -> str:
    return HEADING_NUMBER_RE.sub("", title).strip()


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw_token in TOKEN_RE.findall(text.lower()):
        if contains_cjk(raw_token):
            tokens.extend(tokenize_cjk(raw_token))
        elif raw_token not in EN_STOP_WORDS:
            tokens.append(raw_token)
    return tokens


def contains_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def tokenize_cjk(text: str) -> list[str]:
    normalized = text
    for phrase in QUESTION_STOP_PHRASES:
        normalized = normalized.replace(phrase, " ")
    for char in ("和", "与", "的", "了", "在"):
        normalized = normalized.replace(char, " ")

    tokens: list[str] = []
    for segment in normalized.split():
        tokens.extend(tokenize_cjk_segment(segment))
    return tokens


def tokenize_cjk_segment(segment: str) -> list[str]:
    tokens = [
        phrase
        for phrase in DOMAIN_PHRASES
        if contains_cjk(phrase) and phrase in segment
    ]

    if len(segment) <= 2:
        if segment and segment not in CJK_STOP_WORDS:
            tokens.append(segment)
        return tokens

    tokens.extend(
        token
        for token in (
            segment[index : index + 2] for index in range(len(segment) - 1)
        )
        if token not in CJK_STOP_WORDS
    )
    return tokens


def keyword_score(query: str, chunk: JsonlChunk) -> float:
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return 0.0

    title_tokens = set(tokenize(build_title_search_text(chunk)))
    body_tokens = set(tokenize(build_search_text(chunk)))

    title_overlap = query_tokens & title_tokens
    body_overlap = query_tokens & body_tokens
    return (2.0 * len(title_overlap) + len(body_overlap)) / len(query_tokens)


def keyword_top_k(query: str, chunks: list[JsonlChunk], top_k: int) -> list[RetrievalHit]:
    hits = [
        RetrievalHit(chunk=chunk, score=keyword_score(query, chunk), mode="keyword")
        for chunk in chunks
    ]
    return top_positive(hits, top_k)


def fake_embed_text(text: str, dim: int = 96) -> list[float]:
    vector = [0.0] * dim
    for token in tokenize(text):
        digest = hashlib.md5(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], byteorder="big") % dim
        vector[index] += 1.0
    return normalize(vector)


def normalize(vector: list[float]) -> list[float]:
    norm = sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def embed_chunks(chunks: list[JsonlChunk]) -> list[EmbeddedJsonlChunk]:
    return [
        EmbeddedJsonlChunk(chunk=chunk, vector=fake_embed_text(build_search_text(chunk)))
        for chunk in chunks
    ]


def vector_top_k(
    query: str,
    embedded_chunks: list[EmbeddedJsonlChunk],
    top_k: int,
) -> list[RetrievalHit]:
    query_vector = fake_embed_text(query)
    hits = [
        RetrievalHit(
            chunk=item.chunk,
            score=cosine_similarity(query_vector, item.vector),
            mode="vector",
        )
        for item in embedded_chunks
    ]
    return top_positive(hits, top_k)


def hybrid_top_k(
    query: str,
    chunks: list[JsonlChunk],
    embedded_chunks: list[EmbeddedJsonlChunk],
    top_k: int,
    keyword_weight: float = 0.7,
    vector_weight: float = 0.3,
) -> list[RetrievalHit]:
    keyword_hits = keyword_top_k(query, chunks, top_k=len(chunks))
    vector_hits = vector_top_k(query, embedded_chunks, top_k=len(embedded_chunks))

    merged: dict[str, tuple[JsonlChunk, float, float]] = {}
    for hit in keyword_hits:
        _, _, vector_score = merged.get(hit.chunk.id, (hit.chunk, 0.0, 0.0))
        merged[hit.chunk.id] = (hit.chunk, hit.score, vector_score)

    for hit in vector_hits:
        _, keyword_score, _ = merged.get(hit.chunk.id, (hit.chunk, 0.0, 0.0))
        merged[hit.chunk.id] = (hit.chunk, keyword_score, hit.score)

    hits = [
        RetrievalHit(
            chunk=chunk,
            score=keyword_weight * keyword_score + vector_weight * vector_score,
            mode="hybrid",
        )
        for chunk, keyword_score, vector_score in merged.values()
    ]
    return top_positive(hits, top_k)


def retrieve_top_k(
    query: str,
    chunks: list[JsonlChunk],
    embedded_chunks: list[EmbeddedJsonlChunk],
    top_k: int = 3,
    mode: RetrievalMode = "hybrid",
) -> list[RetrievalHit]:
    if mode == "keyword":
        return keyword_top_k(query, chunks, top_k)
    if mode == "vector":
        return vector_top_k(query, embedded_chunks, top_k)
    return hybrid_top_k(query, chunks, embedded_chunks, top_k)


def top_positive(hits: list[RetrievalHit], top_k: int) -> list[RetrievalHit]:
    hits.sort(
        key=lambda hit: (
            hit.score,
            -(hit.chunk.line_start or 0),
        ),
        reverse=True,
    )
    return [hit for hit in hits if hit.score > 0][:top_k]


def format_hit(index: int, hit: RetrievalHit) -> str:
    heading = " > ".join(hit.chunk.heading_path) or "(no heading)"
    line_range = format_line_range(hit.chunk)
    preview = hit.chunk.text.replace("\n", " ")[:180]
    return (
        f"{index}. score={hit.score:.4f} mode={hit.mode} "
        f"id={hit.chunk.id} type={hit.chunk.chunk_type} lines={line_range}\n"
        f"   heading={heading}\n"
        f"   preview={preview}"
    )


def format_line_range(chunk: JsonlChunk) -> str:
    if chunk.line_start is None or chunk.line_end is None:
        return "unknown"
    return f"{chunk.line_start}-{chunk.line_end}"


def optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def unique_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve TopK chunks from AI-17 JSONL output.")
    parser.add_argument("query", nargs="?", default="epoll 水平触发和边缘触发区别")
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL_PATH)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--mode", choices=["keyword", "vector", "hybrid"], default="hybrid")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunks = load_jsonl_chunks(args.jsonl)
    embedded_chunks = embed_chunks(chunks)
    hits = retrieve_top_k(
        query=args.query,
        chunks=chunks,
        embedded_chunks=embedded_chunks,
        top_k=args.top_k,
        mode=args.mode,
    )

    print(f"loaded_chunks={len(chunks)}")
    print(f"query={args.query}")
    print(f"mode={args.mode}")
    for index, hit in enumerate(hits, start=1):
        print(format_hit(index, hit))


if __name__ == "__main__":
    main()
