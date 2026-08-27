"""Markdown parsing and parent-child chunking for the AI-25B slice."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, Sequence

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
IMAGE_RE = re.compile(r"!\[([^]]*)\]\(([^)\n]+)\)")
TABLE_SEPARATOR_RE = re.compile(
    r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$"
)


class Tokenizer(Protocol):
    name: str

    def encode(self, text: str) -> list[int]: ...

    def decode(self, tokens: list[int]) -> str: ...


class TiktokenTokenizer:
    """Adapter around the project's pinned tiktoken dependency."""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        try:
            import tiktoken
        except ImportError as exc:
            raise RuntimeError(
                "Markdown token chunking requires the dependencies in requirements.txt"
            ) from exc
        self._encoding = tiktoken.get_encoding(encoding_name)
        self.name = encoding_name

    def encode(self, text: str) -> list[int]:
        return self._encoding.encode(text, disallowed_special=())

    def decode(self, tokens: list[int]) -> str:
        return self._encoding.decode(tokens)


class ChunkRole(StrEnum):
    PARENT = "parent"
    CHILD = "child"


@dataclass(frozen=True)
class SourceLine:
    number: int
    text: str


@dataclass(frozen=True)
class MarkdownSection:
    heading_path: tuple[str, ...]
    heading_line: SourceLine | None
    body_lines: tuple[SourceLine, ...]

    @property
    def line_start(self) -> int:
        return self.heading_line.number if self.heading_line else self.body_lines[0].number

    @property
    def line_end(self) -> int:
        return self.body_lines[-1].number if self.body_lines else self.line_start


@dataclass(frozen=True)
class ImageReference:
    alt_text: str
    uri: str
    line_number: int


@dataclass(frozen=True)
class MarkdownUnit:
    kind: str
    text: str
    line_start: int
    line_end: int
    image_references: tuple[ImageReference, ...] = ()

    @property
    def protected(self) -> bool:
        return self.kind in {"code", "table", "image"}


@dataclass(frozen=True)
class IngestedChunk:
    id: str
    revision_id: str
    role: ChunkRole
    parent_chunk_id: str | None
    ordinal: int
    content: str
    content_hash: str
    token_count: int
    tokenizer_name: str
    heading_path: tuple[str, ...]
    source_uri: str
    line_start: int
    line_end: int
    image_references: tuple[ImageReference, ...] = ()

    @property
    def source_location(self) -> str:
        return f"{self.source_uri}:{self.line_start}-{self.line_end}"


@dataclass(frozen=True)
class ParentChildChunks:
    parents: tuple[IngestedChunk, ...]
    children: tuple[IngestedChunk, ...]


def normalize_content(content: str) -> str:
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in normalized.split("\n")).strip()


def stable_content_hash(content: str) -> str:
    return hashlib.sha256(normalize_content(content).encode("utf-8")).hexdigest()


def parse_markdown(markdown: str) -> list[MarkdownSection]:
    """Split on ATX headings while ignoring heading-like text inside fences."""
    sections: list[MarkdownSection] = []
    heading_path: list[str] = []
    heading_line: SourceLine | None = None
    body: list[SourceLine] = []
    fence_char: str | None = None

    def flush() -> None:
        nonlocal body
        if heading_line is not None or any(line.text.strip() for line in body):
            sections.append(MarkdownSection(tuple(heading_path), heading_line, tuple(body)))
        body = []

    for number, text in enumerate(markdown.splitlines(), 1):
        line = SourceLine(number, text)
        fence = FENCE_RE.match(text)
        if fence_char:
            body.append(line)
            if fence and fence.group(1)[0] == fence_char:
                fence_char = None
            continue
        if fence:
            fence_char = fence.group(1)[0]
            body.append(line)
            continue
        heading = HEADING_RE.match(text)
        if heading:
            flush()
            level = len(heading.group(1))
            heading_path = heading_path[: level - 1]
            heading_path.append(heading.group(2).strip())
            heading_line = line
        else:
            body.append(line)
    flush()
    return sections


def _is_table_start(lines: Sequence[SourceLine], index: int) -> bool:
    return (
        index + 1 < len(lines)
        and "|" in lines[index].text
        and TABLE_SEPARATOR_RE.match(lines[index + 1].text) is not None
    )


def _extract_image_references(lines: Sequence[SourceLine]) -> tuple[ImageReference, ...]:
    return tuple(
        ImageReference(match.group(1), match.group(2), line.number)
        for line in lines
        for match in IMAGE_RE.finditer(line.text)
    )


def _section_units(section: MarkdownSection) -> list[MarkdownUnit]:
    lines, units, paragraph = section.body_lines, [], []

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            units.append(
                MarkdownUnit(
                    "text",
                    "\n".join(line.text for line in paragraph).strip(),
                    paragraph[0].number,
                    paragraph[-1].number,
                )
            )
            paragraph = []

    index = 0
    while index < len(lines):
        line = lines[index]
        fence = FENCE_RE.match(line.text)
        if fence:
            flush_paragraph()
            marker = fence.group(1)[0]
            block = [line]
            index += 1
            while index < len(lines):
                block.append(lines[index])
                closing = FENCE_RE.match(lines[index].text)
                index += 1
                if closing and closing.group(1)[0] == marker:
                    break
            units.append(
                MarkdownUnit(
                    "code",
                    "\n".join(x.text for x in block),
                    block[0].number,
                    block[-1].number,
                )
            )
            continue
        if _is_table_start(lines, index):
            flush_paragraph()
            table = [line, lines[index + 1]]
            index += 2
            while index < len(lines) and lines[index].text.strip() and "|" in lines[index].text:
                table.append(lines[index])
                index += 1
            units.append(
                MarkdownUnit(
                    "table",
                    "\n".join(x.text for x in table),
                    table[0].number,
                    table[-1].number,
                    _extract_image_references(table),
                )
            )
            continue
        image_references = _extract_image_references([line])
        if image_references:
            flush_paragraph()
            units.append(
                MarkdownUnit(
                    "image",
                    line.text,
                    line.number,
                    line.number,
                    image_references,
                )
            )
            index += 1
            continue
        if line.text.strip():
            paragraph.append(line)
        else:
            flush_paragraph()
        index += 1
    flush_paragraph()
    return units


def _prefix(section: MarkdownSection) -> str:
    return " > ".join(section.heading_path)


def _join(prefix: str, units: Sequence[MarkdownUnit]) -> str:
    body = "\n\n".join(unit.text for unit in units if unit.text)
    return "\n\n".join(part for part in (prefix, body) if part)


def _collect_image_references(
    units: Sequence[MarkdownUnit],
) -> tuple[ImageReference, ...]:
    return tuple(reference for unit in units for reference in unit.image_references)


def _split_unit(
    unit: MarkdownUnit, *, prefix: str, budget: int, tokenizer: Tokenizer
) -> list[MarkdownUnit]:
    if unit.protected or len(tokenizer.encode(_join(prefix, [unit]))) <= budget:
        return [unit]
    available = budget - len(tokenizer.encode(f"{prefix}\n\n" if prefix else ""))
    if available <= 0:
        raise ValueError("heading path alone exceeds the configured token budget")
    tokens = tokenizer.encode(unit.text)
    parts: list[MarkdownUnit] = []
    for start in range(0, len(tokens), available):
        text = tokenizer.decode(tokens[start : start + available]).strip()
        if text:
            parts.append(MarkdownUnit("text", text, unit.line_start, unit.line_end))
    return parts


def _pack(
    units: Sequence[MarkdownUnit], *, prefix: str, budget: int, tokenizer: Tokenizer
) -> list[list[MarkdownUnit]]:
    expanded = [
        part
        for unit in units
        for part in _split_unit(unit, prefix=prefix, budget=budget, tokenizer=tokenizer)
    ]
    if not expanded:
        return [[]]
    groups: list[list[MarkdownUnit]] = []
    current: list[MarkdownUnit] = []
    for unit in expanded:
        candidate = [*current, unit]
        if current and len(tokenizer.encode(_join(prefix, candidate))) > budget:
            groups.append(current)
            current = [unit]
        else:
            current = candidate
    groups.append(current)
    return groups


def _chunk_id(
    revision_id: str,
    role: ChunkRole,
    ordinal: int,
    parent_id: str | None,
    content_hash: str,
) -> str:
    identity = "\0".join((revision_id, role.value, str(ordinal), parent_id or "", content_hash))
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
    return f"{role.value}-{digest}"


class ParentChildMarkdownChunker:
    """Build parent context chunks and smaller retrieval children."""

    def __init__(
        self,
        *,
        tokenizer: Tokenizer | None = None,
        parent_max_tokens: int = 900,
        child_max_tokens: int = 220,
    ) -> None:
        if parent_max_tokens <= 0 or child_max_tokens <= 0:
            raise ValueError("token budgets must be positive")
        if child_max_tokens > parent_max_tokens:
            raise ValueError("child token budget cannot exceed parent token budget")
        self.tokenizer = tokenizer or TiktokenTokenizer()
        self.parent_max_tokens = parent_max_tokens
        self.child_max_tokens = child_max_tokens

    def chunk(self, markdown: str, *, revision_id: str, source_uri: str) -> ParentChildChunks:
        parents: list[IngestedChunk] = []
        children: list[IngestedChunk] = []
        for section in parse_markdown(markdown):
            prefix = _prefix(section)
            for parent_units in _pack(
                _section_units(section),
                prefix=prefix,
                budget=self.parent_max_tokens,
                tokenizer=self.tokenizer,
            ):
                parent_content = _join(prefix, parent_units)
                parent_hash = stable_content_hash(parent_content)
                parent_ordinal = len(parents)
                parent_id = _chunk_id(
                    revision_id, ChunkRole.PARENT, parent_ordinal, None, parent_hash
                )
                parent_start = parent_units[0].line_start if parent_units else section.line_start
                parent_end = parent_units[-1].line_end if parent_units else section.line_end
                parents.append(
                    IngestedChunk(
                        parent_id, revision_id, ChunkRole.PARENT, None, parent_ordinal,
                        parent_content, parent_hash, len(self.tokenizer.encode(parent_content)),
                        self.tokenizer.name, section.heading_path, source_uri,
                        parent_start, parent_end, _collect_image_references(parent_units),
                    )
                )
                for child_units in _pack(
                    parent_units,
                    prefix=prefix,
                    budget=self.child_max_tokens,
                    tokenizer=self.tokenizer,
                ):
                    child_content = _join(prefix, child_units)
                    child_hash = stable_content_hash(child_content)
                    child_ordinal = len(children)
                    child_start = child_units[0].line_start if child_units else section.line_start
                    child_end = child_units[-1].line_end if child_units else section.line_end
                    children.append(
                        IngestedChunk(
                            _chunk_id(
                                revision_id, ChunkRole.CHILD, child_ordinal,
                                parent_id, child_hash,
                            ),
                            revision_id, ChunkRole.CHILD, parent_id, child_ordinal,
                            child_content, child_hash,
                            len(self.tokenizer.encode(child_content)), self.tokenizer.name,
                            section.heading_path, source_uri, child_start, child_end,
                            _collect_image_references(child_units),
                        )
                    )
        return ParentChildChunks(tuple(parents), tuple(children))
