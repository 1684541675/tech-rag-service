"""Adapter that reuses AI-25 cited generation inside the LangGraph answer node."""
from __future__ import annotations

from collections.abc import Callable

from rag_core.generation import RagAnswerService

from .state import AgentState
from .tools import ControlledAgentTools


def build_rag_answerer(*, tools: ControlledAgentTools, answer_service: RagAnswerService, context_token_budget: int = 1200, max_tokens: int = 600) -> Callable[[AgentState], dict[str, object]]:
    """Build an answer node adapter without duplicating RAG prompt or citation logic."""
    if context_token_budget <= 0 or max_tokens <= 0:
        raise ValueError("context_token_budget and max_tokens must be positive")

    def answer(state: AgentState) -> dict[str, object]:
        ref_ids = [str(context["ref_id"]) for context in state.get("source_context", []) if "ref_id" in context]
        context = tools.generation_context(ref_ids or None)
        return answer_service.answer(
            query=state["user_task"],
            windows=context.windows,
            evidence=context.evidence,
            context_token_budget=context_token_budget,
            max_tokens=max_tokens,
        ).to_dict()

    return answer
