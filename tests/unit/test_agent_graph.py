import unittest

from rag_core.agent.answering import build_rag_answerer
from rag_core.agent.graph import build_graph
from rag_core.agent.state import initial_state
from rag_core.agent.tools import ControlledAgentTools
from rag_core.generation import RagAnswerService, TokenBudgeter
from rag_core.ingestion import ParentChildMarkdownChunker
from rag_core.retrieval import EvidenceDecision, ParentWindowRetriever, RetrievedContext, rrf_fuse
from rag_core.stores import KeywordHit


class Tokenizer:
    name = "character-v1"
    def encode(self, text): return [ord(character) for character in text]
    def decode(self, tokens): return "".join(chr(token) for token in tokens)


class FakeChat:
    def __init__(self): self.calls = 0
    def complete(self, *, messages, model, max_tokens):
        self.calls += 1
        return "ET 模式需要读到 EAGAIN [1]。"


def context(*, sufficient=True):
    chunks = ParentChildMarkdownChunker(tokenizer=Tokenizer(), parent_max_tokens=2000, child_max_tokens=50).chunk("# epoll\n\nedge triggered event loop " * 18, revision_id="rev-1", source_uri="notes.md")
    hits = rrf_fuse({"sparse": [KeywordHit(chunks.children[0].id, "rev-1", 1.0)]}, revision_id="rev-1", limit=1)
    windows = ParentWindowRetriever(parents=chunks.parents, children=chunks.children).fetch(hits)
    return RetrievedContext(windows, EvidenceDecision(sufficient, "sufficient_evidence" if sufficient else "dense_score_below_threshold", 1, 1, 0.91 if sufficient else 0.60, ()))


class AgentGraphTest(unittest.TestCase):
    def test_sufficient_answer_traces_retrieve_inspect_answer(self):
        tools = ControlledAgentTools(run_id="run-1", retrieve_context=lambda _: context())
        graph = build_graph(tools=tools, answerer=lambda state: f"answer with {len(state['source_context'])} source")
        result = graph.invoke(initial_state(run_id="run-1", user_task="epoll ET", max_steps=4))
        self.assertEqual(result["terminal_status"], "completed")
        self.assertEqual(result["answer"], "answer with 1 source")
        self.assertEqual(result["trace"], ["classify:answer", "retrieve:sufficient_evidence", "inspect:1", "answer:completed", "finish:completed"])

    def test_insufficient_evidence_short_circuits_answerer(self):
        tools = ControlledAgentTools(run_id="run-1", retrieve_context=lambda _: context(sufficient=False))
        graph = build_graph(tools=tools, answerer=lambda _: self.fail("answerer must not run"))
        result = graph.invoke(initial_state(run_id="run-1", user_task="MongoDB change stream"))
        self.assertEqual(result["terminal_status"], "no_knowledge")
        self.assertNotIn("answer", result)

    def test_retrieval_failure_ends_without_answer(self):
        tools = ControlledAgentTools(run_id="run-1", retrieve_context=lambda _: (_ for _ in ()).throw(RuntimeError("offline")))
        graph = build_graph(tools=tools, answerer=lambda _: self.fail("answerer must not run"))
        result = graph.invoke(initial_state(run_id="run-1", user_task="epoll"))
        self.assertEqual(result["terminal_status"], "failed")
        self.assertEqual(result["error_code"], "tool_unavailable")

    def test_step_limit_is_enforced_by_graph_nodes(self):
        tools = ControlledAgentTools(run_id="run-1", retrieve_context=lambda _: context())
        result = build_graph(tools=tools).invoke(initial_state(run_id="run-1", user_task="epoll", max_steps=1))
        self.assertEqual(result["terminal_status"], "agent_step_limit")
        self.assertEqual(result["error_code"], "max_steps_exceeded")

    def test_graph_answer_node_reuses_rag_service_and_returns_citations(self):
        tools = ControlledAgentTools(run_id="run-1", retrieve_context=lambda _: context())
        chat = FakeChat()
        service = RagAnswerService(budgeter=TokenBudgeter(Tokenizer()), chat_client=chat)
        graph = build_graph(tools=tools, answerer=build_rag_answerer(tools=tools, answer_service=service, context_token_budget=80))
        result = graph.invoke(initial_state(run_id="run-1", user_task="epoll ET"))
        self.assertEqual(result["terminal_status"], "completed")
        self.assertEqual(result["answer_status"], "ok")
        self.assertEqual(chat.calls, 1)
        self.assertEqual(result["citations"][0]["source_uri"], "notes.md")


if __name__ == "__main__":
    unittest.main()
