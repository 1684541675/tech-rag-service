from __future__ import annotations

import copy
import os
from collections import OrderedDict
from threading import Lock
from time import perf_counter
from typing import Any, Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

from ai18_jsonl_retrieval import EmbeddingProvider, RetrievalMode
from ai20_glm_rag_answer import DEFAULT_GLM_MODEL, answer_query, answer_to_dict


EmbeddingProviderRequest = Literal["auto", "fake", "zhipu"]

app = FastAPI(title="Bagu RAG API")

CACHE_LIMIT = 32
_response_cache: OrderedDict[tuple[object, ...], dict[str, Any]] = OrderedDict()
_cache_lock = Lock()


class BaguRagQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=300)
    top_k: int = Field(default=3, ge=1, le=8)
    mode: RetrievalMode = "hybrid"
    embedding_provider: EmbeddingProviderRequest = "auto"
    use_glm: bool = True
    glm_model: str | None = None
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)
    max_tokens: int = Field(default=800, ge=128, le=2000)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        query = value.strip()
        if not query:
            raise ValueError("query must not be empty")
        return query

    @field_validator("glm_model")
    @classmethod
    def normalize_glm_model(cls, value: str | None) -> str | None:
        if value is None:
            return None
        model = value.strip()
        return model or None


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "service": "bagu-rag",
        "default_glm_model": os.getenv("ZAI_GLM_MODEL") or DEFAULT_GLM_MODEL,
        "zai_api_key_configured": bool(os.getenv("ZAI_API_KEY")),
        "cache_limit": CACHE_LIMIT,
        "cache_size": cache_size(),
    }


@app.post("/agent/query")
def agent_query(request: BaguRagQueryRequest) -> dict[str, object]:
    return run_bagu_rag_query(request)


@app.post("/rag/query")
def rag_query(request: BaguRagQueryRequest) -> dict[str, object]:
    return run_bagu_rag_query(request)


def run_bagu_rag_query(request: BaguRagQueryRequest) -> dict[str, object]:
    start = perf_counter()
    embedding_provider = resolve_embedding_provider(request.embedding_provider)
    glm_model = request.glm_model or os.getenv("ZAI_GLM_MODEL") or DEFAULT_GLM_MODEL
    cache_key = build_cache_key(request, embedding_provider, glm_model)

    cached = get_cached_response(cache_key)
    if cached is not None:
        cached["cache_hit"] = True
        cached["cached_response"] = True
        cached["latency_ms"] = elapsed_ms(start)
        return cached

    try:
        answer = answer_query(
            query=request.query,
            top_k=request.top_k,
            mode=request.mode,
            embedding_provider=embedding_provider,
            use_glm=request.use_glm,
            glm_model=glm_model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        payload = answer_to_dict(answer)
    except RuntimeError as exc:
        payload = build_error_response(
            request=request,
            embedding_provider=embedding_provider,
            glm_model=glm_model,
            error=exc,
            start=start,
        )
    else:
        payload.update(
            {
                "cache_hit": False,
                "cached_response": False,
                "mode": request.mode,
                "embedding_provider": embedding_provider,
                "requested_embedding_provider": request.embedding_provider,
            }
        )

    set_cached_response(cache_key, payload)
    return payload


def resolve_embedding_provider(provider: EmbeddingProviderRequest) -> EmbeddingProvider:
    if provider == "auto":
        return "zhipu" if os.getenv("ZAI_API_KEY") else "fake"
    return provider


def build_cache_key(
    request: BaguRagQueryRequest,
    embedding_provider: EmbeddingProvider,
    glm_model: str,
) -> tuple[object, ...]:
    return (
        request.query,
        request.top_k,
        request.mode,
        embedding_provider,
        request.use_glm,
        glm_model,
        round(request.temperature, 3),
        request.max_tokens,
    )


def get_cached_response(cache_key: tuple[object, ...]) -> dict[str, Any] | None:
    with _cache_lock:
        cached = _response_cache.get(cache_key)
        if cached is None:
            return None
        _response_cache.move_to_end(cache_key)
        return copy.deepcopy(cached)


def set_cached_response(cache_key: tuple[object, ...], payload: dict[str, Any]) -> None:
    with _cache_lock:
        _response_cache[cache_key] = copy.deepcopy(payload)
        _response_cache.move_to_end(cache_key)
        while len(_response_cache) > CACHE_LIMIT:
            _response_cache.popitem(last=False)


def cache_size() -> int:
    with _cache_lock:
        return len(_response_cache)


def build_error_response(
    *,
    request: BaguRagQueryRequest,
    embedding_provider: EmbeddingProvider,
    glm_model: str,
    error: RuntimeError,
    start: float,
) -> dict[str, object]:
    return {
        "query": request.query,
        "answer": f"RAG request failed before answer generation: {error}",
        "status": "retrieval_error",
        "degraded": True,
        "fallback_reason": str(error),
        "model": glm_model if request.use_glm else None,
        "sources": [],
        "diagnostics": {},
        "usage": {},
        "latency_ms": elapsed_ms(start),
        "retrieval_latency_ms": 0,
        "generation_latency_ms": 0,
        "cache_hit": False,
        "cached_response": False,
        "mode": request.mode,
        "embedding_provider": embedding_provider,
        "requested_embedding_provider": request.embedding_provider,
    }


def elapsed_ms(start: float) -> int:
    return int((perf_counter() - start) * 1000)
