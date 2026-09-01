import importlib.util
import unittest

from rag_core.agent import AgentToolError, ControlledAgentTools
from rag_core.ingestion import ParentChildMarkdownChunker
from rag_core.retrieval import EvidenceDecision, ParentWindowRetriever, RetrievedContext, rrf_fuse
from rag_core.stores import KeywordHit


class Tokenizer:
    name = "character-v1"

    def encode(self, text): return [ord(character) for character in text]
    def decode(self, tokens): return "".join(chr(token) for token in tokens)


def sample_windows():
    chunks = ParentChildMarkdownChunker(tokenizer=Tokenizer(), parent_max_tokens=2000, child_max_tokens=50).chunk(
        "# epoll\n\nedge triggered event loop " * 18,
        revision_id="rev-1",
        source_uri="notes.md",
    )
    hits = rrf_fuse({"sparse": [KeywordHit(chunks.children[0].id, "rev-1", 1.0)]}, revision_id="rev-1", limit=1)
    return ParentWindowRetriever(parents=chunks.parents, children=chunks.children).fetch(hits)


def decision(sufficient=True):
    return EvidenceDecision(sufficient, "sufficient_evidence" if sufficient else "dense_score_below_threshold", 1, 1, 0.91 if sufficient else 0.60, ("fixture",))


class ControlledAgentToolsTest(unittest.TestCase):
    def test_sufficient_evidence_creates_run_scoped_ref_and_inspects_source(self):
        tools = ControlledAgentTools(run_id="run-1", retrieve_context=lambda _: RetrievedContext(tuple(sample_windows()), decision()))
        result = tools.retrieve_evidence({"query": "  epoll edge trigger  "})
        self.assertTrue(result["sufficient"])
        self.assertEqual(len(result["refs"]), 1)
        source = tools.inspect_source({"ref_id": result["refs"][0]["ref_id"]})
        self.assertIn("edge triggered", source["content"])
        self.assertEqual(source["citation"]["revision_id"], "rev-1")

    def test_insufficient_evidence_exposes_no_refs_and_cannot_inspect(self):
        tools = ControlledAgentTools(run_id="run-1", retrieve_context=lambda _: RetrievedContext(tuple(sample_windows()), decision(False)))
        result = tools.retrieve_evidence({"query": "MongoDB change stream"})
        self.assertFalse(result["sufficient"])
        self.assertEqual(result["refs"], [])
        with self.assertRaisesRegex(AgentToolError, "来源引用不存在"):
            tools.inspect_source({"ref_id": "ref_from_another_run"})

    def test_invalid_input_and_retrieval_failure_are_controlled(self):
        tools = ControlledAgentTools(run_id="run-1", retrieve_context=lambda _: (_ for _ in ()).throw(RuntimeError("offline")))
        with self.assertRaisesRegex(AgentToolError, "工具参数"):
            tools.retrieve_evidence({"query": "   "})
        with self.assertRaisesRegex(AgentToolError, "检索工具当前不可用"):
            tools.retrieve_evidence({"query": "epoll"})

    @unittest.skipUnless(importlib.util.find_spec("langchain_core"), "LangChain is not installed")
    def test_langchain_structured_tools_bind_the_same_run_scoped_contract(self):
        tools = ControlledAgentTools(run_id="run-1", retrieve_context=lambda _: RetrievedContext(tuple(sample_windows()), decision()))
        retrieve, inspect = tools.as_langchain_tools()
        result = retrieve.invoke({"query": "epoll"})
        source = inspect.invoke({"ref_id": result["refs"][0]["ref_id"]})
        self.assertTrue(result["sufficient"])
        self.assertIn("edge triggered", source["content"])


if __name__ == "__main__":
    unittest.main()
