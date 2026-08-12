from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Any, Iterable, Literal

from ai17_markdown_ingestion import DEFAULT_OUTPUT_PATH as DEFAULT_JSONL_PATH


RetrievalMode = Literal["keyword", "vector", "hybrid"]
EmbeddingProvider = Literal["fake", "zhipu"]

DEFAULT_ZHIPU_MODEL = "embedding-3"
DEFAULT_ZHIPU_DIMENSIONS = 1024
DEFAULT_EMBEDDING_CACHE_PATH = Path(__file__).resolve().parent / "data" / "bagu_embeddings_zhipu.json"
ZHIPU_EMBEDDINGS_URL = "https://open.bigmodel.cn/api/paas/v4/embeddings"
ZHIPU_BATCH_SIZE = 64

TOKEN_RE = re.compile(r"[a-zA-Z0-9_+#.]+|[\u4e00-\u9fff]+")
HEADING_NUMBER_RE = re.compile(r"^[一二三四五六七八九十百零〇两]+\s*、\s*")
ARABIC_COUNT_RE = re.compile(r"(?<![a-zA-Z0-9])([234])\s*次")
TOP_K_VARIANT_RE = re.compile(r"(?<![a-zA-Z0-9_])top\s*(?:\(\s*k\s*\)|k)(?![a-zA-Z0-9_])")

ARABIC_TO_CJK_COUNT = {
    "2": "两",
    "3": "三",
    "4": "四",
}

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

CONCEPT_QUERY_HINTS = (
    "是什么",
    "为什么",
    "有什么区别",
    "区别",
    "关系",
    "原理",
    "分别",
    "有哪些",
    "常见",
    "怎么办",
)

CODE_QUERY_HINTS = (
    "代码",
    "实现",
    "示例",
    "例子",
    "写法",
    "怎么写",
    "如何写",
    "demo",
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

DOMAIN_PHRASE_SET = set(DOMAIN_PHRASES)


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


@dataclass(frozen=True)
class RetrievalDiagnostics:
    query_tokens: list[str]
    core_terms: list[str]
    matched_core_terms: list[str]
    missing_core_terms: list[str]
    top_score: float
    possible_knowledge_gap: bool
    fallback_reason: str | None


@dataclass(frozen=True)
class RetrievalResult:
    hits: list[RetrievalHit]
    diagnostics: RetrievalDiagnostics


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
    for raw_token in TOKEN_RE.findall(normalize_search_text(text.lower())):
        if contains_cjk(raw_token):
            tokens.extend(tokenize_cjk(raw_token))
        elif raw_token not in EN_STOP_WORDS:
            tokens.append(raw_token)
    return tokens


def normalize_search_text(text: str) -> str:
    normalized = TOP_K_VARIANT_RE.sub("topk", text)
    return ARABIC_COUNT_RE.sub(
        lambda match: f"{ARABIC_TO_CJK_COUNT[match.group(1)]}次",
        normalized,
    )


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


def extract_core_terms(query: str) -> list[str]:
    tokens = tokenize(query)
    core_terms = [
        token
        for token in tokens
        if token in DOMAIN_PHRASE_SET
        or token in {"c++", "tcp", "udp", "linux", "mysql", "redis", "epoll"}
        or len(token) >= 3
    ]
    core_terms.extend(infer_query_domain_phrases(query, tokens))
    return unique_preserve_order(core_terms)


def extract_domain_phrases(text: str) -> set[str]:
    normalized = normalize_search_text(text.lower())
    return {phrase for phrase in DOMAIN_PHRASES if phrase in normalized}


def infer_query_domain_phrases(query: str, tokens: list[str] | None = None) -> list[str]:
    normalized = normalize_search_text(query.lower())
    token_set = set(tokens or tokenize(query))
    inferred: list[str] = []

    if "两次" in token_set and ("握手" in token_set or "三次握手" in token_set):
        inferred.append("两次握手")
    if "三次" in token_set and "握手" in token_set:
        inferred.append("三次握手")
    if "四次" in token_set and "挥手" in token_set:
        inferred.append("四次挥手")

    if "缓存" in token_set or "redis" in token_set:
        if "穿透" in token_set:
            inferred.append("缓存穿透")
        if "击穿" in token_set:
            inferred.append("缓存击穿")
        if "雪崩" in token_set:
            inferred.append("缓存雪崩")

    for phrase in DOMAIN_PHRASES:
        if phrase in normalized:
            inferred.append(phrase)

    return unique_preserve_order(inferred)


def is_concept_query(query: str) -> bool:
    normalized = normalize_search_text(query.lower())
    return any(hint in normalized for hint in CONCEPT_QUERY_HINTS)


def is_code_query(query: str) -> bool:
    normalized = normalize_search_text(query.lower())
    return any(hint in normalized for hint in CODE_QUERY_HINTS)


def keyword_score(query: str, chunk: JsonlChunk) -> float:
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return 0.0

    title_text = build_title_search_text(chunk)
    search_text = build_search_text(chunk)

    title_tokens = set(tokenize(title_text))
    body_tokens = set(tokenize(build_search_text(chunk)))
    query_phrases = extract_domain_phrases(query) | set(
        infer_query_domain_phrases(query, list(query_tokens))
    )
    title_phrases = query_phrases & extract_domain_phrases(title_text)
    body_phrases = query_phrases & extract_domain_phrases(search_text)

    title_overlap = query_tokens & title_tokens
    body_overlap = query_tokens & body_tokens
    score = (
        4.0 * len(title_phrases)
        + 2.2 * len(title_overlap)
        + 1.5 * len(body_phrases)
        + len(body_overlap)
    ) / len(query_tokens)
    return apply_chunk_penalty(query, chunk, score)


def apply_chunk_penalty(query: str, chunk: JsonlChunk, score: float) -> float:
    if chunk.chunk_type == "code" and is_concept_query(query) and not is_code_query(query):
        return score * 0.45
    return score


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


def zhipu_embed_texts(
    texts: list[str],
    *,
    model: str = DEFAULT_ZHIPU_MODEL,
    dimensions: int = DEFAULT_ZHIPU_DIMENSIONS,
    api_key: str | None = None,
) -> list[list[float]]:
    if not texts:
        return []

    resolved_api_key = api_key or os.getenv("ZAI_API_KEY")
    if not resolved_api_key:
        raise RuntimeError("ZAI_API_KEY is not set; configure it before using zhipu embeddings.")

    body = json.dumps(
        {
            "model": model,
            "input": texts,
            "dimensions": dimensions,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        ZHIPU_EMBEDDINGS_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {resolved_api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Zhipu embedding request failed: HTTP {exc.code} {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Zhipu embedding request failed: {exc}") from exc

    data = sorted(payload.get("data", []), key=lambda item: item.get("index", 0))
    vectors = [normalize([float(value) for value in item["embedding"]]) for item in data]
    if len(vectors) != len(texts):
        raise RuntimeError(
            f"Zhipu embedding response count mismatch: expected {len(texts)}, got {len(vectors)}"
        )
    return vectors


def zhipu_embed_text(
    text: str,
    *,
    model: str = DEFAULT_ZHIPU_MODEL,
    dimensions: int = DEFAULT_ZHIPU_DIMENSIONS,
) -> list[float]:
    return zhipu_embed_texts([text], model=model, dimensions=dimensions)[0]


def load_embedding_cache(path: Path) -> dict[str, list[float]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    vectors = payload.get("vectors", payload)
    return {
        str(chunk_id): [float(value) for value in vector]
        for chunk_id, vector in vectors.items()
    }


def save_embedding_cache(path: Path, vectors: dict[str, list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "provider": "zhipu",
        "model": DEFAULT_ZHIPU_MODEL,
        "dimensions": DEFAULT_ZHIPU_DIMENSIONS,
        "created_at": int(time.time()),
        "vectors": vectors,
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False)


def embed_chunks(
    chunks: list[JsonlChunk],
    *,
    provider: EmbeddingProvider = "fake",
    cache_path: Path = DEFAULT_EMBEDDING_CACHE_PATH,
    refresh_cache: bool = False,
) -> list[EmbeddedJsonlChunk]:
    if provider == "fake":
        return [
            EmbeddedJsonlChunk(chunk=chunk, vector=fake_embed_text(build_search_text(chunk)))
            for chunk in chunks
        ]
    return embed_chunks_with_zhipu(chunks, cache_path=cache_path, refresh_cache=refresh_cache)


def embed_chunks_with_zhipu(
    chunks: list[JsonlChunk],
    *,
    cache_path: Path = DEFAULT_EMBEDDING_CACHE_PATH,
    refresh_cache: bool = False,
) -> list[EmbeddedJsonlChunk]:
    cached_vectors = {} if refresh_cache else load_embedding_cache(cache_path)
    missing_chunks = [chunk for chunk in chunks if chunk.id not in cached_vectors]

    for start in range(0, len(missing_chunks), ZHIPU_BATCH_SIZE):
        batch = missing_chunks[start : start + ZHIPU_BATCH_SIZE]
        vectors = zhipu_embed_texts([build_search_text(chunk) for chunk in batch])
        for chunk, vector in zip(batch, vectors):
            cached_vectors[chunk.id] = vector

    if missing_chunks or refresh_cache:
        save_embedding_cache(cache_path, cached_vectors)

    return [
        EmbeddedJsonlChunk(chunk=chunk, vector=cached_vectors[chunk.id])
        for chunk in chunks
    ]


def vector_top_k(
    query: str,
    embedded_chunks: list[EmbeddedJsonlChunk],
    top_k: int,
    embedding_provider: EmbeddingProvider = "fake",
) -> list[RetrievalHit]:
    query_vector = (
        fake_embed_text(query)
        if embedding_provider == "fake"
        else zhipu_embed_text(query)
    )
    hits = [
        RetrievalHit(
            chunk=item.chunk,
            score=apply_chunk_penalty(
                query,
                item.chunk,
                cosine_similarity(query_vector, item.vector),
            ),
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
    embedding_provider: EmbeddingProvider = "fake",
    keyword_weight: float = 0.7,
    vector_weight: float = 0.3,
) -> list[RetrievalHit]:
    keyword_hits = keyword_top_k(query, chunks, top_k=len(chunks))
    vector_hits = vector_top_k(
        query,
        embedded_chunks,
        top_k=len(embedded_chunks),
        embedding_provider=embedding_provider,
    )

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
    embedding_provider: EmbeddingProvider = "fake",
) -> list[RetrievalHit]:
    if mode == "keyword":
        return keyword_top_k(query, chunks, top_k)
    if mode == "vector":
        return vector_top_k(query, embedded_chunks, top_k, embedding_provider)
    return hybrid_top_k(query, chunks, embedded_chunks, top_k, embedding_provider)


def retrieve_with_diagnostics(
    query: str,
    chunks: list[JsonlChunk],
    embedded_chunks: list[EmbeddedJsonlChunk],
    top_k: int = 3,
    mode: RetrievalMode = "hybrid",
    embedding_provider: EmbeddingProvider = "fake",
) -> RetrievalResult:
    hits = retrieve_top_k(query, chunks, embedded_chunks, top_k, mode, embedding_provider)
    diagnostics = build_diagnostics(query, hits)
    return RetrievalResult(hits=hits, diagnostics=diagnostics)


def build_diagnostics(query: str, hits: list[RetrievalHit]) -> RetrievalDiagnostics:
    query_tokens = unique_preserve_order(tokenize(query))
    core_terms = extract_core_terms(query)
    matched_core_terms = [
        term for term in core_terms if any(term_matches_hit(term, hit) for hit in hits)
    ]
    missing_core_terms = [
        term for term in core_terms if term not in matched_core_terms
    ]
    top_score = hits[0].score if hits else 0.0
    possible_knowledge_gap, fallback_reason = detect_knowledge_gap(
        query=query,
        core_terms=core_terms,
        matched_core_terms=matched_core_terms,
        top_score=top_score,
        hits=hits,
    )
    return RetrievalDiagnostics(
        query_tokens=query_tokens,
        core_terms=core_terms,
        matched_core_terms=matched_core_terms,
        missing_core_terms=missing_core_terms,
        top_score=top_score,
        possible_knowledge_gap=possible_knowledge_gap,
        fallback_reason=fallback_reason,
    )


def term_matches_hit(term: str, hit: RetrievalHit) -> bool:
    haystack = normalize_search_text(
        "\n".join(
            [
                build_title_search_text(hit.chunk),
                build_search_text(hit.chunk),
            ]
        ).lower()
    )
    return normalize_search_text(term.lower()) in haystack


def detect_knowledge_gap(
    query: str,
    core_terms: list[str],
    matched_core_terms: list[str],
    top_score: float,
    hits: list[RetrievalHit],
) -> tuple[bool, str | None]:
    if not hits:
        return True, "no_positive_hits"

    if top_score < 0.45:
        return True, "top_score_below_threshold"

    if core_terms:
        matched_ratio = len(matched_core_terms) / len(core_terms)
        if matched_ratio < 0.5:
            return True, "core_terms_missing_from_topk"

    if (
        hits[0].chunk.chunk_type == "code"
        and is_concept_query(query)
        and not is_code_query(query)
    ):
        return True, "top_hit_is_code_for_weak_match"

    return False, None


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


def format_diagnostics(diagnostics: RetrievalDiagnostics) -> str:
    return (
        "diagnostics="
        + json.dumps(
            {
                "query_tokens": diagnostics.query_tokens,
                "core_terms": diagnostics.core_terms,
                "matched_core_terms": diagnostics.matched_core_terms,
                "missing_core_terms": diagnostics.missing_core_terms,
                "top_score": round(diagnostics.top_score, 4),
                "possible_knowledge_gap": diagnostics.possible_knowledge_gap,
                "fallback_reason": diagnostics.fallback_reason,
            },
            ensure_ascii=False,
        )
    )


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
    parser.add_argument(
        "--embedding-provider",
        choices=["fake", "zhipu"],
        default="fake",
        help="Use fake hash vectors by default, or Zhipu embedding-3 when ZAI_API_KEY is set.",
    )
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=DEFAULT_EMBEDDING_CACHE_PATH,
        help="Local cache for Zhipu chunk embeddings.",
    )
    parser.add_argument(
        "--refresh-embedding-cache",
        action="store_true",
        help="Regenerate the Zhipu chunk embedding cache.",
    )
    parser.add_argument(
        "--embedding-smoke-test",
        action="store_true",
        help="Only request one query embedding and print its dimension; does not embed chunks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.embedding_smoke_test:
        if args.embedding_provider != "zhipu":
            print("embedding_smoke_test requires --embedding-provider zhipu")
            return
        vector = zhipu_embed_text(args.query)
        print(f"embedding_provider={args.embedding_provider}")
        print(f"query={args.query}")
        print(f"embedding_dim={len(vector)}")
        print(f"embedding_preview={vector[:3]}")
        return

    chunks = load_jsonl_chunks(args.jsonl)
    embedded_chunks = embed_chunks(
        chunks,
        provider=args.embedding_provider,
        cache_path=args.embedding_cache,
        refresh_cache=args.refresh_embedding_cache,
    )
    result = retrieve_with_diagnostics(
        query=args.query,
        chunks=chunks,
        embedded_chunks=embedded_chunks,
        top_k=args.top_k,
        mode=args.mode,
        embedding_provider=args.embedding_provider,
    )

    print(f"loaded_chunks={len(chunks)}")
    print(f"query={args.query}")
    print(f"mode={args.mode}")
    print(f"embedding_provider={args.embedding_provider}")
    print(format_diagnostics(result.diagnostics))
    for index, hit in enumerate(result.hits, start=1):
        print(format_hit(index, hit))


if __name__ == "__main__":
    main()
