"""Serializable state contract for the controlled LangGraph Agent."""
from __future__ import annotations

from typing import Literal, TypedDict


TerminalStatus = Literal[
    "running",
    "completed",
    "no_knowledge",
    "failed",
    "agent_step_limit",
    "unsupported_task",
    "awaiting_approval",
    "rejected",
]


class AgentState(TypedDict, total=False):
    """Only run-scoped, non-secret values permitted to cross graph nodes."""

    run_id: str
    thread_id: str
    user_task: str
    task_kind: Literal["answer", "mastery_update"]
    inspect_enabled: bool
    max_inspections: int
    max_steps: int
    step_count: int
    retrieval: dict[str, object]
    source_context: list[dict[str, object]]
    answer: str
    answer_status: str
    citations: list[dict[str, object]]
    pending_proposal: dict[str, object]
    approval_status: str
    degraded: bool
    terminal_status: TerminalStatus
    error_code: str | None
    trace: list[str]


def initial_state(*, run_id: str, user_task: str, thread_id: str | None = None, inspect_enabled: bool = True, max_inspections: int = 2, max_steps: int = 4) -> AgentState:
    if not run_id.strip() or not user_task.strip() or (thread_id is not None and not thread_id.strip()):
        raise ValueError("run_id, user_task and thread_id must not be empty")
    if max_inspections < 0 or max_steps <= 0:
        raise ValueError("max_inspections must be non-negative and max_steps must be positive")
    return {
        "run_id": run_id,
        "thread_id": thread_id or run_id,
        "user_task": user_task.strip(),
        "inspect_enabled": inspect_enabled,
        "max_inspections": max_inspections,
        "max_steps": max_steps,
        "step_count": 0,
        "source_context": [],
        "citations": [],
        "terminal_status": "running",
        "approval_status": "not_required",
        "error_code": None,
        "trace": [],
    }
