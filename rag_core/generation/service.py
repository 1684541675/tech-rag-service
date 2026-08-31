"""Token-bounded, cited answer generation for a revision-scoped RAG result."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Callable, Protocol, Sequence

from rag_core.ingestion.markdown import Tokenizer
from rag_core.retrieval import EvidenceDecision, ParentWindow

ZHIPU_CHAT_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"


@dataclass(frozen=True)
class Citation:
    index: int
    revision_id: str
    parent_chunk_id: str
    source_uri: str
    heading_path: tuple[str, ...]
    line_start: int
    line_end: int


@dataclass(frozen=True)
class ContextBlock:
    citation: Citation
    content: str
    token_count: int
    truncated: bool


@dataclass(frozen=True)
class RagAnswer:
    answer: str
    status: str
    degraded: bool
    fallback_reason: str | None
    model: str | None
    citations: tuple[Citation, ...]
    context_tokens: int

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "citations": [asdict(citation) for citation in self.citations]}


class ChatClient(Protocol):
    def complete(self, *, messages: list[dict[str, str]], model: str, max_tokens: int) -> str: ...


class TokenBudgeter:
    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer

    def select(self, windows: Sequence[ParentWindow], *, token_budget: int) -> tuple[ContextBlock, ...]:
        if token_budget <= 0:
            raise ValueError("token_budget must be positive")
        selected: list[ContextBlock] = []
        remaining = token_budget
        for index, window in enumerate(windows, 1):
            parent = window.parent
            tokens = self.tokenizer.encode(parent.content)
            if not tokens or remaining <= 0:
                break
            take = min(len(tokens), remaining)
            selected.append(ContextBlock(
                Citation(index, parent.revision_id, parent.id, parent.source_uri, parent.heading_path, parent.line_start, parent.line_end),
                self.tokenizer.decode(tokens[:take]), take, take < len(tokens),
            ))
            remaining -= take
        return tuple(selected)


class ZhipuChatClient:
    """Minimal stdlib GLM client; key is read only at invocation time."""
    def __init__(self, *, api_key: str | None = None, timeout_seconds: int = 45) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def complete(self, *, messages: list[dict[str, str]], model: str, max_tokens: int) -> str:
        api_key = self.api_key or os.getenv("ZAI_API_KEY")
        if not api_key:
            raise RuntimeError("ZAI_API_KEY is not configured")
        request = urllib.request.Request(
            ZHIPU_CHAT_URL,
            data=json.dumps({"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.2}, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError) as exc:
            raise RuntimeError(f"GLM request failed: {exc}") from exc
        content = str(payload.get("choices", [{}])[0].get("message", {}).get("content", "")).strip()
        if not content:
            raise RuntimeError("GLM response has no answer content")
        return content


class RagAnswerService:
    """Generates an answer from already revision-scoped parent windows."""
    def __init__(self, *, budgeter: TokenBudgeter, chat_client: ChatClient, model: str = "glm-4-flash") -> None:
        self.budgeter, self.chat_client, self.model = budgeter, chat_client, model

    def answer(self, *, query: str, windows: Sequence[ParentWindow], evidence: EvidenceDecision | None = None, context_token_budget: int = 1200, max_tokens: int = 600) -> RagAnswer:
        if not query.strip():
            raise ValueError("query must not be empty")
        if evidence is None:
            return RagAnswer("当前请求尚未完成证据评估，不能生成回答。", "no_knowledge", True, "evidence_not_assessed", None, (), 0)
        if not evidence.sufficient:
            return RagAnswer("当前知识库没有足够的可引用资料来回答这个问题。", "no_knowledge", True, evidence.reason, None, (), 0)
        blocks = self.budgeter.select(windows, token_budget=context_token_budget)
        citations = tuple(block.citation for block in blocks)
        if not blocks:
            return RagAnswer("没有检索到可引用的资料。", "no_knowledge", True, "no_context", None, citations, 0)
        messages = _messages(query, blocks)
        try:
            text = self.chat_client.complete(messages=messages, model=self.model, max_tokens=max_tokens)
        except RuntimeError as exc:
            return RagAnswer(_summary(blocks, str(exc)), "llm_unavailable", True, str(exc), self.model, citations, sum(block.token_count for block in blocks))
        return RagAnswer(text, "ok", False, None, self.model, citations, sum(block.token_count for block in blocks))


def _messages(query: str, blocks: Sequence[ContextBlock]) -> list[dict[str, str]]:
    context = "\n\n".join(f"[{block.citation.index}] {block.content}" for block in blocks)
    return [
        {"role": "system", "content": "仅依据给定资料回答；资料不足时说明不足；关键结论标记对应 [编号]。"},
        {"role": "user", "content": f"问题：{query}\n\n资料：\n{context}"},
    ]


def _summary(blocks: Sequence[ContextBlock], reason: str) -> str:
    excerpts = "\n".join(f"[{block.citation.index}] {block.content}" for block in blocks)
    return f"GLM 不可用，已降级为可追溯检索摘要。原因：{reason}\n{excerpts}"
