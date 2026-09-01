import json
import tempfile
import unittest
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from rag_core.agent.approval import MasteryUpdateStore
from rag_core.agent.graph import build_graph
from rag_core.agent.state import initial_state
from rag_core.agent.tools import ControlledAgentTools
from tests.unit.test_agent_graph import context


class AgentApprovalTest(unittest.TestCase):
    def _graph(self, path: Path):
        tools = ControlledAgentTools(run_id="run-approval", retrieve_context=lambda _: context())
        return build_graph(tools=tools, approval_store=MasteryUpdateStore(record_path=path), checkpointer=InMemorySaver())

    def test_reject_does_not_write_and_approve_appends_one_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mastery_updates.jsonl"
            graph = self._graph(path)
            config = {"configurable": {"thread_id": "thread-approval"}}
            paused = graph.invoke(initial_state(run_id="run-approval", thread_id="thread-approval", user_task="记录掌握 epoll ET"), config=config)
            self.assertIn("__interrupt__", paused)
            proposal = paused["__interrupt__"][0].value["proposal"]
            rejected = graph.invoke(Command(resume={"action": "reject"}), config=config)
            self.assertEqual(rejected["terminal_status"], "rejected")
            self.assertFalse(path.exists())

            graph = self._graph(path)
            config = {"configurable": {"thread_id": "thread-approve"}}
            graph.invoke(initial_state(run_id="run-approval", thread_id="thread-approve", user_task="记录掌握 epoll ET"), config=config)
            approved = graph.invoke(Command(resume={"action": "approve"}), config=config)
            self.assertEqual(approved["terminal_status"], "completed")
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["status"], "approved")


if __name__ == "__main__":
    unittest.main()
