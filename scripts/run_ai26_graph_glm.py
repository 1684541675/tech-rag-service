"""Run the Stage-2 LangGraph with AI-25 cited GLM generation."""
from __future__ import annotations

import sys
from uuid import uuid4

from rag_core.agent import ControlledAgentTools, build_rag_answerer, initial_state
from rag_core.agent.graph import build_graph
from rag_core.generation import RagAnswerService, TokenBudgeter, ZhipuChatClient
from rag_core.ingestion import TiktokenTokenizer
from scripts.query_active_bagu_core import build_active_retrieve_context


def main() -> None:
    task = sys.argv[1] if len(sys.argv) > 1 else "解释 epoll 边缘触发时为什么必须读到 EAGAIN，并给出来源。"
    run_id = str(uuid4())
    tools = ControlledAgentTools(run_id=run_id, retrieve_context=build_active_retrieve_context())
    service = RagAnswerService(budgeter=TokenBudgeter(TiktokenTokenizer()), chat_client=ZhipuChatClient(timeout_seconds=300))
    graph = build_graph(tools=tools, answerer=build_rag_answerer(tools=tools, answer_service=service))
    result = graph.invoke(initial_state(run_id=run_id, user_task=task))
    print(f"run_id={run_id} status={result['terminal_status']} answer_status={result.get('answer_status')} degraded={result.get('degraded')}")
    print(f"trace={result['trace']}")
    print(f"citations={[(citation.get('heading_path'), citation.get('line_start'), citation.get('line_end')) for citation in result.get('citations', [])]}")
    if result.get("answer"):
        print(result["answer"])
    elif result["terminal_status"] == "no_knowledge":
        print(f"知识库证据不足，未调用 GLM 生成。reason={result.get('evidence_reason')}")
    else:
        print(result.get("error_code") or "未产生回答")


if __name__ == "__main__":
    main()
