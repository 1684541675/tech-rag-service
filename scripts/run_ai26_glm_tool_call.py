"""Run one real, evidence-grounded GLM Function Calling trace for AI-26."""
from __future__ import annotations

import sys
from uuid import uuid4

from rag_core.agent import ControlledAgentTools, GlmToolCallingAgent
from scripts.query_active_bagu_core import build_active_retrieve_context


def main() -> None:
    task = sys.argv[1] if len(sys.argv) > 1 else "解释 epoll 边缘触发时为什么必须读到 EAGAIN，并给出来源。"
    run_id = str(uuid4())
    tools = ControlledAgentTools(run_id=run_id, retrieve_context=build_active_retrieve_context())
    result = GlmToolCallingAgent(tools=tools).run(user_task=task)
    print(f"run_id={run_id} status={result.status} model={result.model}")
    print(f"tool_calls={result.tool_calls} usage={dict(result.usage or {})}")
    print(result.answer or "当前知识库没有足够证据，或模型未遵守工具协议。")


if __name__ == "__main__":
    main()
