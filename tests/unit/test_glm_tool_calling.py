import unittest

from rag_core.agent import ControlledAgentTools, GlmToolCallingAgent
from rag_core.ingestion import ParentChildMarkdownChunker
from rag_core.retrieval import EvidenceDecision, ParentWindowRetriever, RetrievedContext, rrf_fuse
from rag_core.stores import KeywordHit


class Tokenizer:
    name = "character-v1"
    def encode(self, text): return [ord(character) for character in text]
    def decode(self, tokens): return "".join(chr(token) for token in tokens)


def context(*, sufficient=True):
    chunks = ParentChildMarkdownChunker(tokenizer=Tokenizer(), parent_max_tokens=2000, child_max_tokens=50).chunk("# epoll\n\nedge triggered event loop " * 18, revision_id="rev-1", source_uri="notes.md")
    hits = rrf_fuse({"sparse": [KeywordHit(chunks.children[0].id, "rev-1", 1.0)]}, revision_id="rev-1", limit=1)
    windows = ParentWindowRetriever(parents=chunks.parents, children=chunks.children).fetch(hits)
    return RetrievedContext(windows, EvidenceDecision(sufficient, "sufficient_evidence" if sufficient else "dense_score_below_threshold", 1, 1, 0.91 if sufficient else 0.60, ()))


class GlmToolCallingAgentTest(unittest.TestCase):
    def test_real_protocol_shape_executes_retrieval_then_returns_final_answer(self):
        responses = iter([
            {"choices": [{"message": {"role": "assistant", "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "retrieve_evidence", "arguments": '{"query":"epoll"}'}}]}}]},
            {"choices": [{"message": {"role": "assistant", "content": "ET 应读到 EAGAIN [1]"}}], "usage": {"total_tokens": 12}},
        ])
        agent = GlmToolCallingAgent(tools=ControlledAgentTools(run_id="run-1", retrieve_context=lambda _: context()), transport=lambda _: next(responses))
        result = agent.run(user_task="ET 为什么读到 EAGAIN？")
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.tool_calls, ("retrieve_evidence",))
        self.assertIn("EAGAIN", result.answer)

    def test_insufficient_evidence_stops_before_second_model_request(self):
        calls = []
        def transport(_):
            calls.append(1)
            return {"choices": [{"message": {"role": "assistant", "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "retrieve_evidence", "arguments": '{"query":"MongoDB"}'}}]}}]}
        agent = GlmToolCallingAgent(tools=ControlledAgentTools(run_id="run-1", retrieve_context=lambda _: context(sufficient=False)), transport=transport)
        result = agent.run(user_task="MongoDB change stream")
        self.assertEqual(result.status, "no_knowledge")
        self.assertEqual(len(calls), 1)

    def test_model_answer_without_retrieval_is_rejected(self):
        agent = GlmToolCallingAgent(tools=ControlledAgentTools(run_id="run-1", retrieve_context=lambda _: context()), transport=lambda _: {"choices": [{"message": {"role": "assistant", "content": "编造回答"}}]})
        self.assertEqual(agent.run(user_task="epoll").status, "agent_protocol_error")

    def test_model_unavailable_fails_closed(self):
        def unavailable(_):
            raise RuntimeError("HTTP 429")
        agent = GlmToolCallingAgent(tools=ControlledAgentTools(run_id="run-1", retrieve_context=lambda _: context()), transport=unavailable)
        result = agent.run(user_task="epoll")
        self.assertEqual(result.status, "llm_unavailable")
        self.assertIsNone(result.answer)


if __name__ == "__main__":
    unittest.main()
