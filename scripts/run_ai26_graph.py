"""Run the deterministic Stage-2 LangGraph path against the active RAG Core."""
from __future__ import annotations

import sys
from uuid import uuid4

from rag_core.agent.graph import build_graph
from rag_core.agent.state import initial_state
from rag_core.agent.tools import ControlledAgentTools
from scripts.query_active_bagu_core import build_active_retrieve_context


def main() -> None:
    task = sys.argv[1] if len(sys.argv) > 1 else "解释 epoll 边缘触发时为什么必须读到 EAGAIN。"
    run_id = str(uuid4())
    graph = build_graph(tools=ControlledAgentTools(run_id=run_id, retrieve_context=build_active_retrieve_context()))
    result = graph.invoke(initial_state(run_id=run_id, user_task=task))
    print(f"run_id={run_id} status={result['terminal_status']} steps={result['step_count']}")
    print(f"trace={result['trace']}")
    if result["terminal_status"] == "no_knowledge":
        reason = result.get("retrieval", {}).get("reason", "unknown")
        print(f"当前知识库没有足够的可引用资料来回答这个问题。reason={reason}")
    else:
        print(result.get("answer") or result.get("error_code") or "no answer")


if __name__ == "__main__":
    main()
