"""Start the manual human-in-the-loop API for AI-26 development testing."""
from __future__ import annotations

from pathlib import Path
import sys


# Allow ``python scripts/serve_ai26_agent.py`` from the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn
from langgraph.checkpoint.memory import InMemorySaver

from rag_core.agent.api import create_agent_app
from rag_core.agent.approval import MasteryUpdateStore
from rag_core.agent.answering import build_rag_answerer
from rag_core.agent.graph import build_graph
from rag_core.agent.tools import ControlledAgentTools
from rag_core.generation import RagAnswerService, TokenBudgeter, ZhipuChatClient
from rag_core.ingestion import TiktokenTokenizer
from scripts.query_active_bagu_core import build_active_retrieve_context


RECORD_PATH = PROJECT_ROOT / "data" / "mastery_updates.jsonl"
STORE = MasteryUpdateStore(record_path=RECORD_PATH)


def make_graph(run_id: str):
    tools = ControlledAgentTools(run_id=run_id, retrieve_context=build_active_retrieve_context())
    answer_service = RagAnswerService(
        budgeter=TokenBudgeter(TiktokenTokenizer()),
        chat_client=ZhipuChatClient(timeout_seconds=300),
    )
    return build_graph(
        tools=tools,
        answerer=build_rag_answerer(tools=tools, answer_service=answer_service),
        approval_store=STORE,
        checkpointer=InMemorySaver(),
    )


app = create_agent_app(make_graph=make_graph)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8016)
