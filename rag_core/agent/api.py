"""Minimal HTTP protocol for the Agent's approval/resume boundary."""
from __future__ import annotations

from collections.abc import Callable
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException
from langgraph.types import Command
from pydantic import BaseModel, Field, field_validator

from .state import initial_state
from .trace import LoggingTraceSink, TraceSink, emit, task_fingerprint


class AgentRunRequest(BaseModel):
    task: str = Field(min_length=1, max_length=500)
    thread_id: str | None = Field(default=None, max_length=128)

    @field_validator("task")
    @classmethod
    def strip_task(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("task must not be empty")
        return value


class ApprovalRequest(BaseModel):
    action: str = Field(pattern="^(approve|reject)$")


GraphFactory = Callable[[str], object]


def create_agent_app(*, make_graph: GraphFactory, trace_sink: TraceSink | None = None) -> FastAPI:
    """Create an Agent-only API without changing the existing RAG API contract.

    The registry is deliberately process-local: a source ref is valid only for
    the live run that issued it.  A multi-process deployment must replace this
    with authenticated durable run storage before claiming resumability.
    """
    app = FastAPI(title="evidence-grounded-interview-agent", version="0.1.0")
    pending_graphs: dict[str, object] = {}
    sink = trace_sink or LoggingTraceSink()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "evidence-grounded-interview-agent"}

    @app.post("/agent/run")
    def run(request: AgentRunRequest, x_request_id: str | None = Header(default=None, max_length=128)) -> dict[str, object]:
        run_id = uuid4().hex
        thread_id = request.thread_id or uuid4().hex
        request_id = x_request_id or uuid4().hex
        emit(sink, name="agent.run.started", request_id=request_id, run_id=run_id, fields={"thread_id": thread_id, **task_fingerprint(request.task)})
        graph = make_graph(run_id)
        config = {"configurable": {"thread_id": thread_id}}
        result = graph.invoke(initial_state(run_id=run_id, thread_id=thread_id, user_task=request.task), config=config)
        if "__interrupt__" in result:
            pending_graphs[thread_id] = graph
            response = {"status": "awaiting_approval", "run_id": run_id, "thread_id": thread_id, "request_id": request_id, "interrupt": result["__interrupt__"][0].value}
        else:
            response = _public_result(result, run_id=run_id, thread_id=thread_id, request_id=request_id)
        emit(sink, name="agent.run.finished", request_id=request_id, run_id=run_id, fields={"status": response["status"], "error_code": response.get("error_code")})
        return response

    @app.post("/agent/{thread_id}/approval")
    def approval(thread_id: str, request: ApprovalRequest, x_request_id: str | None = Header(default=None, max_length=128)) -> dict[str, object]:
        request_id = x_request_id or uuid4().hex
        graph = pending_graphs.get(thread_id)
        if graph is None:
            emit(sink, name="agent.approval.missing", request_id=request_id, run_id=None, fields={"thread_id": thread_id})
            raise HTTPException(status_code=404, detail="待审批运行不存在、已过期或不属于当前服务进程。")
        emit(sink, name="agent.approval.started", request_id=request_id, run_id=None, fields={"thread_id": thread_id, "action": request.action})
        result = graph.invoke(Command(resume={"action": request.action}), config={"configurable": {"thread_id": thread_id}})
        pending_graphs.pop(thread_id, None)
        run_id = str(result.get("run_id", ""))
        response = _public_result(result, run_id=run_id, thread_id=thread_id, request_id=request_id)
        emit(sink, name="agent.approval.finished", request_id=request_id, run_id=run_id, fields={"thread_id": thread_id, "status": response["status"], "action": request.action})
        return response

    return app


def _public_result(result: dict[str, object], *, run_id: str, thread_id: str, request_id: str) -> dict[str, object]:
    return {
        "status": result.get("terminal_status"),
        "run_id": run_id,
        "thread_id": thread_id,
        "request_id": request_id,
        "answer": result.get("answer"),
        "answer_status": result.get("answer_status"),
        "citations": result.get("citations", []),
        "approval_status": result.get("approval_status"),
        "trace": result.get("trace", []),
        "error_code": result.get("error_code"),
    }
