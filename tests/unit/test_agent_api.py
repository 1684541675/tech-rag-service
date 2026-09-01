import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from rag_core.agent.api import create_agent_app
from rag_core.agent.approval import MasteryUpdateStore
from rag_core.agent.graph import build_graph
from rag_core.agent.tools import ControlledAgentTools
from tests.unit.test_agent_graph import context


class AgentApiTest(unittest.TestCase):
    def test_run_then_approval_resumes_the_same_thread(self):
        with tempfile.TemporaryDirectory() as directory:
            record_path = Path(directory) / "updates.jsonl"

            def make_graph(run_id):
                return build_graph(
                    tools=ControlledAgentTools(run_id=run_id, retrieve_context=lambda _: context()),
                    approval_store=MasteryUpdateStore(record_path=record_path),
                    checkpointer=InMemorySaver(),
                )

            client = TestClient(create_agent_app(make_graph=make_graph))
            started = client.post("/agent/run", json={"task": "记录掌握 epoll ET", "thread_id": "api-thread"})
            self.assertEqual(started.status_code, 200)
            self.assertEqual(started.json()["status"], "awaiting_approval")
            finished = client.post("/agent/api-thread/approval", json={"action": "approve"})
            self.assertEqual(finished.status_code, 200)
            self.assertEqual(finished.json()["status"], "completed")
            self.assertTrue(record_path.exists())


if __name__ == "__main__":
    unittest.main()
