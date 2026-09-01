"""Run-scoped, read-only tools for the evidence-grounded interview Agent.

This module deliberately does not implement a graph or call an LLM.  It
adapts the existing revision-bound ``retrieve_context`` boundary into narrow
tool contracts, preserving the AI-25 evidence gate as the authority that
allows or blocks generation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Sequence
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from rag_core.generation import Citation
from rag_core.retrieval import ParentWindow, RetrievedContext


class RetrieveEvidenceInput(BaseModel):
    """Public schema for the only retrieval input exposed to a model."""

    query: str = Field(min_length=1, max_length=500)

    @field_validator("query")
    @classmethod
    def strip_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be empty")
        return value


class InspectSourceInput(BaseModel):
    """Public schema for source inspection; paths and revisions stay private."""

    ref_id: str = Field(min_length=1, max_length=128)


class AgentToolError(RuntimeError):
    """A stable, safe-to-return failure from a controlled tool."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class EvidenceRef:
    """A server-created reference that is only valid within one Agent run."""

    ref_id: str
    citation: Citation
    window: ParentWindow


class ControlledAgentTools:
    """Expose retrieval and source reading without widening RAG Core access.

    ``retrieve_context`` must already select the active revision and apply the
    existing hybrid retrieval plus evidence gate.  This adapter does not accept
    a revision ID, source URI, file path, SQL, or arbitrary store handle.
    """

    def __init__(self, *, run_id: str, retrieve_context: Callable[[str], RetrievedContext]) -> None:
        if not run_id.strip():
            raise ValueError("run_id must not be empty")
        self.run_id = run_id
        self._retrieve_context = retrieve_context
        self._refs: dict[str, EvidenceRef] = {}
        self._last_context: RetrievedContext | None = None

    def retrieve_evidence(self, raw_input: RetrieveEvidenceInput | dict[str, object]) -> dict[str, object]:
        request = self._validate(RetrieveEvidenceInput, raw_input)
        try:
            context = self._retrieve_context(request.query)
        except Exception as exc:
            raise AgentToolError("tool_unavailable", "检索工具当前不可用，无法生成资料性回答。") from exc
        self._last_context = context
        self._refs.clear()

        decision = context.evidence
        response: dict[str, object] = {
            "sufficient": decision.sufficient,
            "reason": decision.reason,
            "diagnostics": list(decision.diagnostics),
            "fused_hit_count": decision.fused_hit_count,
            "window_count": decision.window_count,
            "max_dense_score": decision.max_dense_score,
            "refs": [],
        }
        if not decision.sufficient:
            return response

        refs = [self._register(window, index) for index, window in enumerate(context.windows, 1)]
        response["refs"] = [self._public_ref(ref) for ref in refs]
        return response

    def inspect_source(self, raw_input: InspectSourceInput | dict[str, object]) -> dict[str, object]:
        request = self._validate(InspectSourceInput, raw_input)
        ref = self._refs.get(request.ref_id)
        if ref is None:
            raise AgentToolError("invalid_source_ref", "来源引用不存在、已过期或不属于当前运行。")
        citation = ref.citation
        return {
            "ref_id": ref.ref_id,
            "content": ref.window.parent.content,
            "citation": asdict(citation),
            "heading_path": list(citation.heading_path),
            "line_start": citation.line_start,
            "line_end": citation.line_end,
        }

    def generation_context(self, ref_ids: Sequence[str] | None = None) -> RetrievedContext:
        """Return only current-run, server-side context for the answer adapter.

        This is intentionally not a LangChain tool and is never exposed to a
        model.  It lets the graph reuse AI-25 generation without re-querying or
        widening the public source-reading contract.
        """
        if self._last_context is None:
            raise AgentToolError("evidence_not_retrieved", "回答前必须先完成检索。")
        if not ref_ids:
            return self._last_context
        windows: list[ParentWindow] = []
        seen: set[str] = set()
        for ref_id in ref_ids:
            ref = self._refs.get(ref_id)
            if ref is None:
                raise AgentToolError("invalid_source_ref", "来源引用不存在、已过期或不属于当前运行。")
            if ref.window.parent.id not in seen:
                windows.append(ref.window)
                seen.add(ref.window.parent.id)
        return RetrievedContext(tuple(windows), self._last_context.evidence)

    def as_langchain_tools(self) -> list[object]:
        """Create LangChain structured tools after project dependencies exist.

        The lazy import keeps deterministic unit tests runnable before optional
        framework installation, while production graph code receives ordinary
        schema-validated LangChain tools.
        """
        try:
            from langchain_core.tools import StructuredTool
        except ImportError as exc:
            raise RuntimeError("Install requirements.txt before binding LangChain tools") from exc
        return [
            StructuredTool.from_function(
                func=lambda query: self.retrieve_evidence({"query": query}),
                name="retrieve_evidence",
                description="检索当前知识库中的可引用证据；证据不足时不得生成资料性答案。",
                args_schema=RetrieveEvidenceInput,
            ),
            StructuredTool.from_function(
                func=lambda ref_id: self.inspect_source({"ref_id": ref_id}),
                name="inspect_source",
                description="读取本次运行中 retrieve_evidence 返回的来源上下文。",
                args_schema=InspectSourceInput,
            ),
        ]

    @staticmethod
    def _validate(model: type[BaseModel], raw_input: BaseModel | dict[str, object]) -> BaseModel:
        try:
            return raw_input if isinstance(raw_input, model) else model.model_validate(raw_input)
        except Exception as exc:
            raise AgentToolError("tool_validation_error", "工具参数不符合约束。") from exc

    def _register(self, window: ParentWindow, index: int) -> EvidenceRef:
        parent = window.parent
        ref = EvidenceRef(
            ref_id=f"ref_{uuid4().hex}",
            citation=Citation(index, parent.revision_id, parent.id, parent.source_uri, parent.heading_path, parent.line_start, parent.line_end),
            window=window,
        )
        self._refs[ref.ref_id] = ref
        return ref

    @staticmethod
    def _public_ref(ref: EvidenceRef) -> dict[str, object]:
        citation = ref.citation
        return {
            "ref_id": ref.ref_id,
            "citation": asdict(citation),
            "heading_path": list(citation.heading_path),
            "line_start": citation.line_start,
            "line_end": citation.line_end,
        }
