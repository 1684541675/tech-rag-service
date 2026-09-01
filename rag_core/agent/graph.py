"""Explicit LangGraph state machine for the evidence-grounded Agent."""
from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .approval import ApprovalError, MasteryUpdateStore
from .state import AgentState
from .tools import AgentToolError, ControlledAgentTools

Answerer = Callable[[AgentState], str | dict[str, object]]


def build_graph(*, tools: ControlledAgentTools, answerer: Answerer | None = None, approval_store: MasteryUpdateStore | None = None, checkpointer=None):
    """Build a bounded graph whose only data access is ``ControlledAgentTools``.

    The answerer is injected so deterministic tests do not call an LLM.  In a
    later slice it will be a cited GLM answer adapter; the graph boundary and
    gate routing remain unchanged.
    """
    graph = StateGraph(AgentState)
    graph.add_node("classify", _classify)
    graph.add_node("retrieve", lambda state: _retrieve(state, tools))
    graph.add_node("inspect", lambda state: _inspect(state, tools))
    graph.add_node("answer", lambda state: _answer(state, answerer or _default_answer))
    graph.add_node("propose", lambda state: _propose(state, approval_store))
    graph.add_node("finish", _finish)
    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", _after_classify, {"retrieve": "retrieve", "finish": "finish"})
    graph.add_conditional_edges("retrieve", _after_retrieve, {"inspect": "inspect", "answer": "answer", "propose": "propose", "finish": "finish"})
    graph.add_conditional_edges("inspect", _after_inspect, {"answer": "answer", "propose": "propose", "finish": "finish"})
    graph.add_edge("answer", "finish")
    graph.add_edge("propose", "finish")
    graph.add_edge("finish", END)
    return graph.compile(checkpointer=checkpointer)


def _classify(state: AgentState) -> AgentState:
    step, limited = _advance(state, "classify")
    if limited is not None:
        return limited
    task = state["user_task"]
    kind = "mastery_update" if any(token in task for token in ("记录掌握", "学习记录", "掌握情况")) else "answer"
    return {"step_count": step, "task_kind": kind, "trace": [*state["trace"], f"classify:{kind}"]}


def _retrieve(state: AgentState, tools: ControlledAgentTools) -> AgentState:
    step, limited = _advance(state, "retrieve")
    if limited is not None:
        return limited
    try:
        retrieval = tools.retrieve_evidence({"query": state["user_task"]})
    except AgentToolError as exc:
        return {"step_count": step, "terminal_status": "failed", "error_code": exc.code, "trace": [*state["trace"], f"retrieve:{exc.code}"]}
    status = "running" if bool(retrieval.get("sufficient")) else "no_knowledge"
    return {"step_count": step, "retrieval": retrieval, "terminal_status": status, "trace": [*state["trace"], f"retrieve:{retrieval['reason']}"]}


def _inspect(state: AgentState, tools: ControlledAgentTools) -> AgentState:
    step, limited = _advance(state, "inspect")
    if limited is not None:
        return limited
    refs = state.get("retrieval", {}).get("refs", [])
    if not isinstance(refs, list):
        return {"step_count": step, "terminal_status": "failed", "error_code": "invalid_retrieval_contract", "trace": [*state["trace"], "inspect:invalid_retrieval_contract"]}
    contexts: list[dict[str, object]] = []
    try:
        for ref in refs[:state["max_inspections"]]:
            if isinstance(ref, dict):
                contexts.append(tools.inspect_source({"ref_id": ref["ref_id"]}))
    except (KeyError, AgentToolError) as exc:
        return {"step_count": step, "terminal_status": "failed", "error_code": getattr(exc, "code", "invalid_source_ref"), "trace": [*state["trace"], "inspect:failed"]}
    return {"step_count": step, "source_context": contexts, "trace": [*state["trace"], f"inspect:{len(contexts)}"]}


def _answer(state: AgentState, answerer: Answerer) -> AgentState:
    step, limited = _advance(state, "answer")
    if limited is not None:
        return limited
    try:
        result = answerer(state)
    except Exception:
        return {"step_count": step, "terminal_status": "failed", "error_code": "answer_unavailable", "trace": [*state["trace"], "answer:unavailable"]}
    if isinstance(result, dict):
        status = str(result.get("status", "ok"))
        terminal = "no_knowledge" if status == "no_knowledge" else "completed"
        return {
            "step_count": step,
            "answer": str(result.get("answer", "")),
            "answer_status": status,
            "citations": list(result.get("citations", [])),
            "degraded": bool(result.get("degraded", False)),
            "terminal_status": terminal,
            "trace": [*state["trace"], f"answer:{status}"],
        }
    return {"step_count": step, "answer": result, "answer_status": "summary", "terminal_status": "completed", "trace": [*state["trace"], "answer:completed"]}


def _finish(state: AgentState) -> AgentState:
    return {"trace": [*state["trace"], f"finish:{state['terminal_status']}"]}


def _after_classify(state: AgentState) -> Literal["retrieve", "finish"]:
    if state["terminal_status"] != "running":
        return "finish"
    return "retrieve"


def _after_retrieve(state: AgentState) -> Literal["inspect", "answer", "propose", "finish"]:
    if state["terminal_status"] != "running":
        return "finish"
    if state.get("inspect_enabled") and state.get("max_inspections", 0) > 0 and state.get("retrieval", {}).get("refs"):
        return "inspect"
    return "propose" if state.get("task_kind") == "mastery_update" else "answer"


def _after_inspect(state: AgentState) -> Literal["answer", "propose", "finish"]:
    if state["terminal_status"] != "running":
        return "finish"
    return "propose" if state.get("task_kind") == "mastery_update" else "answer"


def _propose(state: AgentState, store: MasteryUpdateStore | None) -> AgentState:
    step, limited = _advance(state, "propose")
    if limited is not None:
        return limited
    if store is None:
        return {"step_count": step, "terminal_status": "failed", "error_code": "approval_unavailable", "trace": [*state["trace"], "propose:unavailable"]}
    citations = [context.get("citation", {}) for context in state.get("source_context", [])]
    try:
        proposal = store.propose(run_id=state["run_id"], thread_id=state["thread_id"], topic=state["user_task"], mastered=True, note="等待人工确认的学习掌握度更新。", citations=citations)
        decision = interrupt({"kind": "mastery_update_approval", "proposal": proposal.public()})
        result = store.resolve(proposal_id=proposal.proposal_id, run_id=state["run_id"], thread_id=state["thread_id"], action=str(decision.get("action", "")))
    except ApprovalError as exc:
        return {"step_count": step, "terminal_status": "failed", "error_code": exc.code, "trace": [*state["trace"], f"propose:{exc.code}"]}
    status = "completed" if result["status"] == "approved" else "rejected"
    return {"step_count": step, "pending_proposal": proposal.public(), "approval_status": result["status"], "terminal_status": status, "trace": [*state["trace"], f"propose:{result['status']}"]}


def _advance(state: AgentState, node: str) -> tuple[int, AgentState | None]:
    next_step = state["step_count"] + 1
    if next_step > state["max_steps"]:
        return next_step, {"step_count": next_step, "terminal_status": "agent_step_limit", "error_code": "max_steps_exceeded", "trace": [*state["trace"], f"{node}:max_steps_exceeded"]}
    return next_step, None


def _default_answer(state: AgentState) -> str:
    citations = [context.get("citation", {}) for context in state.get("source_context", [])]
    labels = [f"[{index}] {citation.get('source_uri', 'unknown')}:{citation.get('line_start', '?')}-{citation.get('line_end', '?')}" for index, citation in enumerate(citations, 1)]
    return "已完成受证据约束的检索。来源：" + ("；".join(labels) if labels else "未展开来源上下文")
