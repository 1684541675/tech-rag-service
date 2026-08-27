import unittest
from pathlib import Path

from rag_core.ingestion import (
    ParentChildMarkdownChunker,
    TiktokenTokenizer,
    stable_content_hash,
)


FIXTURE = Path(__file__).parents[1] / "fixtures" / "parent_child_sample.md"


class CharacterTokenizer:
    """Deterministic offline tokenizer for exact boundary tests."""

    name = "character-v1"

    def encode(self, text: str) -> list[int]:
        return [ord(character) for character in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


class ParentChildMarkdownChunkingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.markdown = FIXTURE.read_text(encoding="utf-8")
        self.chunker = ParentChildMarkdownChunker(
            tokenizer=CharacterTokenizer(),
            parent_max_tokens=2_000,
            child_max_tokens=230,
        )

    def _chunk(self):
        return self.chunker.chunk(
            self.markdown,
            revision_id="revision-fixture",
            source_uri="tests/fixtures/parent_child_sample.md",
        )

    def test_fixture_has_stable_parent_child_and_source_relationships(self) -> None:
        first = self._chunk()
        second = self._chunk()

        self.assertEqual(first, second)
        self.assertEqual(len(first.parents), 3)
        self.assertEqual(
            [parent.heading_path for parent in first.parents],
            [
                ("Backend Notes",),
                ("Backend Notes", "epoll"),
                ("Backend Notes", "Reactor"),
            ],
        )
        parent_by_id = {parent.id: parent for parent in first.parents}
        self.assertTrue(first.children)
        child_counts = {
            parent.id: sum(child.parent_chunk_id == parent.id for child in first.children)
            for parent in first.parents
        }
        self.assertTrue(any(count > 1 for count in child_counts.values()))
        for child in first.children:
            self.assertIn(child.parent_chunk_id, parent_by_id)
            parent = parent_by_id[child.parent_chunk_id]
            self.assertEqual(child.heading_path, parent.heading_path)
            self.assertEqual(child.source_uri, parent.source_uri)
            self.assertEqual(child.token_count, len(CharacterTokenizer().encode(child.content)))
            self.assertLessEqual(parent.line_start, child.line_start)
            self.assertGreaterEqual(parent.line_end, child.line_end)

    def test_code_block_and_table_are_never_split(self) -> None:
        result = self._chunk()
        code = """```cpp
while (read(fd, buffer, sizeof(buffer)) > 0) {
    consume(buffer);
}
```"""
        table = """| Mode | Application rule |
| --- | --- |
| LT | Readiness may be reported again |
| ET | Drain nonblocking I/O until EAGAIN |"""

        self.assertEqual(sum(code in child.content for child in result.children), 1)
        self.assertEqual(sum(table in child.content for child in result.children), 1)
        for child in result.children:
            self.assertIn(child.content.count("```"), (0, 2))

    def test_markdown_image_is_preserved_as_traceable_metadata(self) -> None:
        result = self._chunk()
        chunks_with_images = [
            child for child in result.children if child.image_references
        ]

        self.assertEqual(len(chunks_with_images), 1)
        image = chunks_with_images[0].image_references[0]
        self.assertEqual(image.alt_text, "image-20260827203704467")
        self.assertEqual(
            image.uri,
            r"C:\Users\1684541675\AppData\Roaming\Typora"
            r"\typora-user-images\image-20260827203704467.png",
        )
        self.assertEqual(image.line_number, 32)
        self.assertIn(
            f"![{image.alt_text}]({image.uri})",
            chunks_with_images[0].content,
        )
        parent = next(
            parent
            for parent in result.parents
            if parent.id == chunks_with_images[0].parent_chunk_id
        )
        self.assertEqual(
            parent.image_references,
            chunks_with_images[0].image_references,
        )

    def test_pinned_tokenizer_produces_expected_fixture_budgets(self) -> None:
        result = ParentChildMarkdownChunker(
            tokenizer=TiktokenTokenizer("cl100k_base"),
            parent_max_tokens=900,
            child_max_tokens=220,
        ).chunk(
            self.markdown,
            revision_id="revision-fixture",
            source_uri="tests/fixtures/parent_child_sample.md",
        )

        self.assertEqual(
            [chunk.token_count for chunk in result.parents],
            [20, 98, 320],
        )
        self.assertEqual(
            [chunk.token_count for chunk in result.children],
            [20, 98, 220, 106],
        )
        self.assertTrue(all(chunk.token_count <= 220 for chunk in result.children))

    def test_long_text_splits_children_but_keeps_one_parent(self) -> None:
        markdown = "# Long section\n\n" + ("bounded context " * 80)
        chunker = ParentChildMarkdownChunker(
            tokenizer=CharacterTokenizer(),
            parent_max_tokens=2_000,
            child_max_tokens=120,
        )
        result = chunker.chunk(markdown, revision_id="rev-long", source_uri="long.md")

        self.assertEqual(len(result.parents), 1)
        self.assertGreater(len(result.children), 1)
        self.assertTrue(all(child.token_count <= 120 for child in result.children))
        self.assertTrue(
            all(child.parent_chunk_id == result.parents[0].id for child in result.children)
        )

    def test_content_hash_is_stable_across_platform_line_endings(self) -> None:
        self.assertEqual(
            stable_content_hash("alpha  \r\nbeta\r\n"),
            stable_content_hash("alpha\nbeta"),
        )
        self.assertNotEqual(
            stable_content_hash("alpha\nbeta"),
            stable_content_hash("alpha\ngamma"),
        )

    def test_invalid_token_budgets_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ParentChildMarkdownChunker(
                tokenizer=CharacterTokenizer(),
                parent_max_tokens=100,
                child_max_tokens=101,
            )


if __name__ == "__main__":
    unittest.main()
