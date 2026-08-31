import unittest

from fastapi.testclient import TestClient

from rag_core.api import create_app
from rag_core.generation import RagAnswerService, TokenBudgeter
from rag_core.ingestion import ParentChildMarkdownChunker
from rag_core.retrieval import EvidenceDecision, ParentWindowRetriever, RetrievedContext, rrf_fuse
from rag_core.stores import KeywordHit


class Tokenizer:
    name = "character-v1"
    def encode(self, text): return [ord(character) for character in text]
    def decode(self, tokens): return "".join(chr(token) for token in tokens)


class FakeChat:
    def __init__(self, error=None): self.error, self.calls = error, 0
    def complete(self, *, messages, model, max_tokens):
        self.calls += 1
        if self.error: raise RuntimeError(self.error)
        return "epoll answers are grounded [1]"


def windows():
    chunks = ParentChildMarkdownChunker(tokenizer=Tokenizer(), parent_max_tokens=2000, child_max_tokens=50).chunk("# epoll\n\n" + "edge triggered event loop " * 18, revision_id="rev-1", source_uri="notes.md")
    hits = rrf_fuse({"sparse": [KeywordHit(chunks.children[0].id, "rev-1", 1.0)]}, revision_id="rev-1", limit=1)
    return ParentWindowRetriever(parents=chunks.parents, children=chunks.children).fetch(hits)


def evidence(*, sufficient=True, reason="sufficient_evidence"):
    return EvidenceDecision(sufficient, reason, 1, 1, 0.9 if sufficient else 0.6, ())


class GenerationAndApiTest(unittest.TestCase):
    def test_budget_truncates_context_and_citation_remains_traceable(self):
        service = RagAnswerService(budgeter=TokenBudgeter(Tokenizer()), chat_client=FakeChat())
        answer = service.answer(query="what is epoll", windows=windows(), evidence=evidence(), context_token_budget=30)
        self.assertEqual(answer.status, "ok")
        self.assertEqual(answer.context_tokens, 30)
        self.assertEqual(answer.citations[0].revision_id, "rev-1")
        self.assertEqual(answer.citations[0].source_uri, "notes.md")

    def test_missing_glm_degrades_to_cited_summary(self):
        service = RagAnswerService(budgeter=TokenBudgeter(Tokenizer()), chat_client=FakeChat("ZAI_API_KEY is not configured"))
        answer = service.answer(query="what is epoll", windows=windows(), evidence=evidence(), context_token_budget=50)
        self.assertEqual(answer.status, "llm_unavailable")
        self.assertTrue(answer.degraded)
        self.assertIn("[1]", answer.answer)

    def test_fastapi_protocol_layer_returns_citations(self):
        service = RagAnswerService(budgeter=TokenBudgeter(Tokenizer()), chat_client=FakeChat())
        client = TestClient(create_app(answer_service=service, retrieve_context=lambda _: RetrievedContext(tuple(windows()), evidence())))
        response = client.post("/rag/query", json={"query": "epoll", "context_token_budget": 80})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["citations"][0]["parent_chunk_id"], windows()[0].parent.id)

    def test_insufficient_evidence_skips_glm_and_returns_no_knowledge(self):
        chat = FakeChat()
        service = RagAnswerService(budgeter=TokenBudgeter(Tokenizer()), chat_client=chat)
        answer = service.answer(query="Redis distributed lock", windows=windows(), evidence=evidence(sufficient=False, reason="dense_score_below_threshold"))
        self.assertEqual(answer.status, "no_knowledge")
        self.assertTrue(answer.degraded)
        self.assertEqual(answer.fallback_reason, "dense_score_below_threshold")
        self.assertEqual(answer.citations, ())
        self.assertEqual(chat.calls, 0)


if __name__ == "__main__": unittest.main()
