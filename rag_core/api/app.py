"""FastAPI protocol layer; retrieval orchestration remains injected."""
from __future__ import annotations

from typing import Callable, Sequence

from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

from rag_core.generation import RagAnswerService
from rag_core.retrieval import ParentWindow


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    context_token_budget: int = Field(default=1200, ge=64, le=8000)
    max_tokens: int = Field(default=600, ge=64, le=2000)

    @field_validator("query")
    @classmethod
    def strip_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be empty")
        return value


def create_app(*, answer_service: RagAnswerService, retrieve_windows: Callable[[str], Sequence[ParentWindow]]) -> FastAPI:
    app = FastAPI(title="tech-rag-core", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "service": "tech-rag-core"}

    @app.post("/rag/query")
    def query(request: QueryRequest) -> dict[str, object]:
        windows = retrieve_windows(request.query)
        return answer_service.answer(query=request.query, windows=windows, context_token_budget=request.context_token_budget, max_tokens=request.max_tokens).to_dict()

    return app
