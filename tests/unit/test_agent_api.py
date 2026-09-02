import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from rag_core.agent.api import create_agent_app
from rag_core.agent.approval import MasteryUpdateStore
from rag_core.agent.graph import build_graph
from rag_core.agent.trace import InMemoryTraceSink
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

    def test_request_id_and_safe_structured_trace_cover_success_and_no_knowledge(self):
        trace = InMemoryTraceSink()

        def make_graph(run_id):
            return build_graph(tools=ControlledAgentTools(run_id=run_id, retrieve_context=lambda _: context()))

        client = TestClient(create_agent_app(make_graph=make_graph, trace_sink=trace))
        response = client.post("/agent/run", json={"task": "epoll ET 的工作方式"}, headers={"X-Request-ID": "request-test-1"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["request_id"], "request-test-1")
        self.assertEqual(trace.events[0].name, "agent.run.started")
        self.assertEqual(trace.events[0].fields["task_length"], len("epoll ET 的工作方式"))
        self.assertNotIn("epoll", str(trace.events[0].fields))
        self.assertEqual(trace.events[-1].fields["status"], "completed")

        def no_knowledge_graph(run_id):
            return build_graph(tools=ControlledAgentTools(run_id=run_id, retrieve_context=lambda _: context(sufficient=False)))

        response = TestClient(create_agent_app(make_graph=no_knowledge_graph, trace_sink=trace)).post("/agent/run", json={"task": "MongoDB change stream"})
        self.assertEqual(response.json()["status"], "no_knowledge")
        self.assertEqual(trace.events[-1].fields["status"], "no_knowledge")

    def test_health_endpoint_is_available_without_external_stores(self):
        client = TestClient(create_agent_app(make_graph=lambda _: self.fail("health must not build graph")))
        self.assertEqual(client.get("/health").json()["status"], "ok")


if __name__ == "__main__":
    unittest.main()
