from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from ai18_jsonl_retrieval import (
    DEFAULT_EMBEDDING_CACHE_PATH,
    DEFAULT_JSONL_PATH,
    EmbeddingProvider,
    RetrievalHit,
    RetrievalMode,
    RetrievalResult,
    embed_chunks,
    format_line_range,
    load_jsonl_chunks,
    retrieve_with_diagnostics,
)


DEFAULT_GLM_MODEL = "glm-4-flash"
ZHIPU_CHAT_COMPLETIONS_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

AnswerStatus = Literal[
    "ok",
    "retrieval_empty",
    "knowledge_gap",
    "retrieval_summary",
    "llm_unavailable",
    "llm_error",
]


@dataclass(frozen=True)
class RagSource:
    index: int
    chunk_id: str
    source: str
    heading_path: list[str]
    line_range: str
    chunk_type: str
    score: float
    retrieval_mode: RetrievalMode
    preview: str


@dataclass(frozen=True)
class RagAnswer:
    query: str
    answer: str
    status: AnswerStatus
    degraded: bool
    fallback_reason: str | None
    model: str | None
    sources: list[RagSource]
    diagnostics: dict[str, Any]
    usage: dict[str, Any]
    latency_ms: int
    retrieval_latency_ms: int
    generation_latency_ms: int


def answer_query(
    query: str,
    *,
    jsonl_path: Path = DEFAULT_JSONL_PATH,
    top_k: int = 3,
    mode: RetrievalMode = "hybrid",
    embedding_provider: EmbeddingProvider = "fake",
    embedding_cache_path: Path = DEFAULT_EMBEDDING_CACHE_PATH,
    refresh_embedding_cache: bool = False,
    use_glm: bool = True,
    glm_model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 800,
    timeout: int = 60,
) -> RagAnswer:
    total_start = perf_counter()
    model = glm_model or os.getenv("ZAI_GLM_MODEL") or DEFAULT_GLM_MODEL

    retrieval_start = perf_counter()
    chunks = load_jsonl_chunks(jsonl_path)
    embedded_chunks = embed_chunks(
        chunks,
        provider=embedding_provider,
        cache_path=embedding_cache_path,
        refresh_cache=refresh_embedding_cache,
    )
    retrieval = retrieve_with_diagnostics(
        query=query,
        chunks=chunks,
        embedded_chunks=embedded_chunks,
        top_k=top_k,
        mode=mode,
        embedding_provider=embedding_provider,
    )
    retrieval_latency_ms = elapsed_ms(retrieval_start)

    sources = build_sources(retrieval.hits)
    diagnostics = build_diagnostics_payload(retrieval)

    if not retrieval.hits:
        return build_degraded_answer(
            query=query,
            retrieval=retrieval,
            sources=sources,
            diagnostics=diagnostics,
            status="retrieval_empty",
            reason="no_hits",
            model=None,
            total_start=total_start,
            retrieval_latency_ms=retrieval_latency_ms,
        )

    if retrieval.diagnostics.possible_knowledge_gap:
        return build_degraded_answer(
            query=query,
            retrieval=retrieval,
            sources=sources,
            diagnostics=diagnostics,
            status="knowledge_gap",
            reason=retrieval.diagnostics.fallback_reason or "possible_knowledge_gap",
            model=None,
            total_start=total_start,
            retrieval_latency_ms=retrieval_latency_ms,
        )

    if not use_glm:
        return build_degraded_answer(
            query=query,
            retrieval=retrieval,
            sources=sources,
            diagnostics=diagnostics,
            status="retrieval_summary",
            reason="glm_disabled",
            model=None,
            total_start=total_start,
            retrieval_latency_ms=retrieval_latency_ms,
        )

    generation_start = perf_counter()
    try:
        content, usage = call_glm_chat(
            messages=build_rag_messages(query, retrieval.hits),
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    except RuntimeError as exc:
        status: AnswerStatus = (
            "llm_unavailable" if "ZAI_API_KEY is not set" in str(exc) else "llm_error"
        )
        return build_degraded_answer(
            query=query,
            retrieval=retrieval,
            sources=sources,
            diagnostics=diagnostics,
            status=status,
            reason=str(exc),
            model=model,
            total_start=total_start,
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=elapsed_ms(generation_start),
        )

    return RagAnswer(
        query=query,
        answer=content,
        status="ok",
        degraded=False,
        fallback_reason=None,
        model=model,
        sources=sources,
        diagnostics=diagnostics,
        usage=usage,
        latency_ms=elapsed_ms(total_start),
        retrieval_latency_ms=retrieval_latency_ms,
        generation_latency_ms=elapsed_ms(generation_start),
    )


def build_rag_messages(query: str, hits: list[RetrievalHit]) -> list[dict[str, str]]:
    context = "\n\n".join(format_context_item(index, hit) for index, hit in enumerate(hits, 1))
    return [
        {
            "role": "system",
            "content": (
                "你是一个面向 C++ 后端实习面试的 RAG 助手。"
                "只能根据给定资料回答，不要编造资料中没有的信息。"
                "如果资料不足，直接说明缺少什么。"
                "回答要先讲本质，再给面试表达，并在关键结论后标注来源编号，如 [1]。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"问题：\n{query}\n\n"
                f"检索资料：\n{context}\n\n"
                "请基于检索资料给出简洁、可用于面试复习的中文回答。"
            ),
        },
    ]


def call_glm_chat(
    *,
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    api_key: str | None = None,
) -> tuple[str, dict[str, Any]]:
    resolved_api_key = api_key or os.getenv("ZAI_API_KEY")
    if not resolved_api_key:
        raise RuntimeError("ZAI_API_KEY is not set; configure it before calling GLM.")

    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        ZHIPU_CHAT_COMPLETIONS_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {resolved_api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GLM chat request failed: HTTP {exc.code} {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"GLM chat request failed: {exc}") from exc

    choices = payload.get("choices", [])
    if not choices:
        raise RuntimeError(f"GLM chat response has no choices: {payload}")

    message = choices[0].get("message", {})
    content = str(message.get("content", "")).strip()
    if not content:
        raise RuntimeError(f"GLM chat response has empty content: {payload}")
    return content, dict(payload.get("usage", {}))


def build_degraded_answer(
    *,
    query: str,
    retrieval: RetrievalResult,
    sources: list[RagSource],
    diagnostics: dict[str, Any],
    status: AnswerStatus,
    reason: str,
    model: str | None,
    total_start: float,
    retrieval_latency_ms: int,
    generation_latency_ms: int = 0,
) -> RagAnswer:
    return RagAnswer(
        query=query,
        answer=build_retrieval_summary(query, retrieval, reason),
        status=status,
        degraded=True,
        fallback_reason=reason,
        model=model,
        sources=sources,
        diagnostics=diagnostics,
        usage={},
        latency_ms=elapsed_ms(total_start),
        retrieval_latency_ms=retrieval_latency_ms,
        generation_latency_ms=generation_latency_ms,
    )


def build_retrieval_summary(
    query: str,
    retrieval: RetrievalResult,
    reason: str,
) -> str:
    if not retrieval.hits:
        return (
            f"问题「{query}」没有检索到可用资料。"
            f"降级原因：{reason}。建议补充知识库内容或降低 gap 阈值后再试。"
        )

    lines = [
        f"当前未生成 GLM 回答，已降级为 TopK 检索摘要。降级原因：{reason}。",
        "可参考资料：",
    ]
    for index, hit in enumerate(retrieval.hits, start=1):
        heading = " > ".join(hit.chunk.heading_path) or hit.chunk.title or "(no heading)"
        preview = compact_text(hit.chunk.text, 180)
        lines.append(
            f"[{index}] {heading} | type={hit.chunk.chunk_type} "
            f"score={hit.score:.4f} lines={format_line_range(hit.chunk)}\n{preview}"
        )
    return "\n\n".join(lines)


def format_context_item(index: int, hit: RetrievalHit, max_chars: int = 1200) -> str:
    heading = " > ".join(hit.chunk.heading_path) or hit.chunk.title or "(no heading)"
    return (
        f"[{index}] source={hit.chunk.source}; heading={heading}; "
        f"lines={format_line_range(hit.chunk)}; type={hit.chunk.chunk_type}; "
        f"score={hit.score:.4f}\n"
        f"{compact_text(hit.chunk.text, max_chars)}"
    )


def build_sources(hits: list[RetrievalHit]) -> list[RagSource]:
    return [
        RagSource(
            index=index,
            chunk_id=hit.chunk.id,
            source=hit.chunk.source,
            heading_path=hit.chunk.heading_path,
            line_range=format_line_range(hit.chunk),
            chunk_type=hit.chunk.chunk_type,
            score=round(hit.score, 4),
            retrieval_mode=hit.mode,
            preview=compact_text(hit.chunk.text, 220),
        )
        for index, hit in enumerate(hits, start=1)
    ]


def build_diagnostics_payload(retrieval: RetrievalResult) -> dict[str, Any]:
    diagnostics = retrieval.diagnostics
    return {
        "query_tokens": diagnostics.query_tokens,
        "core_terms": diagnostics.core_terms,
        "matched_core_terms": diagnostics.matched_core_terms,
        "missing_core_terms": diagnostics.missing_core_terms,
        "top_score": round(diagnostics.top_score, 4),
        "possible_knowledge_gap": diagnostics.possible_knowledge_gap,
        "fallback_reason": diagnostics.fallback_reason,
    }


def compact_text(text: str, max_chars: int) -> str:
    compacted = " ".join(text.split())
    if len(compacted) <= max_chars:
        return compacted
    return compacted[: max_chars - 3] + "..."


def elapsed_ms(start: float) -> int:
    return int((perf_counter() - start) * 1000)


def answer_to_dict(answer: RagAnswer) -> dict[str, Any]:
    return asdict(answer)


def print_human_readable(answer: RagAnswer) -> None:
    print(f"query={answer.query}")
    print(f"status={answer.status}")
    print(f"degraded={answer.degraded}")
    print(f"fallback_reason={answer.fallback_reason}")
    print(f"model={answer.model}")
    print(f"latency_ms={answer.latency_ms}")
    print()
    print(answer.answer)
    print()
    print("sources:")
    for source in answer.sources:
        print(
            f"[{source.index}] score={source.score:.4f} type={source.chunk_type} "
            f"lines={source.line_range} heading={' > '.join(source.heading_path)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a GLM RAG answer from bagu JSONL chunks.")
    parser.add_argument("query", nargs="?", default="epoll 水平触发和边缘触发有什么区别")
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL_PATH)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--mode", choices=["keyword", "vector", "hybrid"], default="hybrid")
    parser.add_argument(
        "--embedding-provider",
        choices=["fake", "zhipu"],
        default="fake",
        help="Use fake vectors by default, or Zhipu embedding-3 when ZAI_API_KEY is set.",
    )
    parser.add_argument("--embedding-cache", type=Path, default=DEFAULT_EMBEDDING_CACHE_PATH)
    parser.add_argument("--refresh-embedding-cache", action="store_true")
    parser.add_argument("--glm-model", default=os.getenv("ZAI_GLM_MODEL") or DEFAULT_GLM_MODEL)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--no-glm", action="store_true", help="Only return retrieval summary.")
    parser.add_argument("--json", action="store_true", help="Print the full response as JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started_at = int(time.time())
    answer = answer_query(
        query=args.query,
        jsonl_path=args.jsonl,
        top_k=args.top_k,
        mode=args.mode,
        embedding_provider=args.embedding_provider,
        embedding_cache_path=args.embedding_cache,
        refresh_embedding_cache=args.refresh_embedding_cache,
        use_glm=not args.no_glm,
        glm_model=args.glm_model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )

    if args.json:
        payload = answer_to_dict(answer)
        payload["created_at"] = started_at
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    print_human_readable(answer)


if __name__ == "__main__":
    main()
