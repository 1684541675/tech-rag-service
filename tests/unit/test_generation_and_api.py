import unittest

from fastapi.testclient import TestClient

from rag_core.api import create_app
from rag_core.generation import RagAnswerService, TokenBudgeter
from rag_core.ingestion import ParentChildMarkdownChunker
from rag_core.retrieval import ParentWindowRetriever, rrf_fuse
from rag_core.stores import KeywordHit


class Tokenizer:
    name = "character-v1"
    def encode(self, text): return [ord(character) for character in text]
    def decode(self, tokens): return "".join(chr(token) for token in tokens)


class FakeChat:
    def __init__(self, error=None): self.error = error
    def complete(self, *, messages, model, max_tokens):
        if self.error: raise RuntimeError(self.error)
        return "epoll answers are grounded [1]"


def windows():
    chunks = ParentChildMarkdownChunker(tokenizer=Tokenizer(), parent_max_tokens=2000, child_max_tokens=50).chunk("# epoll\n\n" + "edge triggered event loop " * 18, revision_id="rev-1", source_uri="notes.md")
    hits = rrf_fuse({"sparse": [KeywordHit(chunks.children[0].id, "rev-1", 1.0)]}, revision_id="rev-1", limit=1)
    return ParentWindowRetriever(parents=chunks.parents, children=chunks.children).fetch(hits)


class GenerationAndApiTest(unittest.TestCase):
    def test_budget_truncates_context_and_citation_remains_traceable(self):
        service = RagAnswerService(budgeter=TokenBudgeter(Tokenizer()), chat_client=FakeChat())
        answer = service.answer(query="what is epoll", windows=windows(), context_token_budget=30)
        self.assertEqual(answer.status, "ok")
        self.assertEqual(answer.context_tokens, 30)
        self.assertEqual(answer.citations[0].revision_id, "rev-1")
        self.assertEqual(answer.citations[0].source_uri, "notes.md")

    def test_missing_glm_degrades_to_cited_summary(self):
        service = RagAnswerService(budgeter=TokenBudgeter(Tokenizer()), chat_client=FakeChat("ZAI_API_KEY is not configured"))
        answer = service.answer(query="what is epoll", windows=windows(), context_token_budget=50)
        self.assertEqual(answer.status, "llm_unavailable")
        self.assertTrue(answer.degraded)
        self.assertIn("[1]", answer.answer)

    def test_fastapi_protocol_layer_returns_citations(self):
        service = RagAnswerService(budgeter=TokenBudgeter(Tokenizer()), chat_client=FakeChat())
        client = TestClient(create_app(answer_service=service, retrieve_windows=lambda _: windows()))
        response = client.post("/rag/query", json={"query": "epoll", "context_token_budget": 80})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["citations"][0]["parent_chunk_id"], windows()[0].parent.id)


if __name__ == "__main__": unittest.main()
