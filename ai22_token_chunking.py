"""Token-budget Markdown chunking experiment.

This module is intentionally independent from ai17-ai21.  It reuses ai17's
Markdown parsing and JSONL schema, but changes only the text-chunk policy:
text is bounded by a real tokenizer budget and can retain a small token
overlap.  Tables and fenced blocks remain whole, as they do in ai17.

Run after installing the optional local dependency:
    ./.venv/Scripts/python.exe -m pip install tiktoken
    ./.venv/Scripts/python.exe ai22_token_chunking.py --max-tokens 450 --overlap-tokens 60
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

from ai17_markdown_ingestion import (
    DEFAULT_INPUT_PATH,
    DEFAULT_SOURCE,
    IngestedChunk,
    ParsedElement,
    Section,
    make_chunk,
    parse_section_elements,
    read_markdown_lines,
    split_sections,
    summarize,
    unique_preserve_order,
    write_jsonl,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "bagu_chunks_token.jsonl"
DEFAULT_ENCODING = "cl100k_base"
SENTENCE_BREAK_RE = re.compile(r"(?<=[。！？!?；;])\s+|\n+")


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...

    def decode(self, tokens: list[int]) -> str: ...


class TiktokenTokenizer:
    """Thin adapter so the chunking logic is not coupled to tiktoken APIs."""

    def __init__(self, encoding_name: str = DEFAULT_ENCODING) -> None:
        try:
            import tiktoken
        except ImportError as exc:
            raise RuntimeError(
                "Token-level chunking needs tiktoken. Install it in this project's "
                "virtual environment first: .\\.venv\\Scripts\\python.exe -m pip "
                "install tiktoken"
            ) from exc
        self._encoding = tiktoken.get_encoding(encoding_name)
        self.encoding_name = encoding_name

    def encode(self, text: str) -> list[int]:
        return self._encoding.encode(text, disallowed_special=())

    def decode(self, tokens: list[int]) -> str:
        return self._encoding.decode(tokens)


@dataclass(frozen=True)
class TokenTextUnit:
    text: str
    line_start: int
    line_end: int
    image_urls: tuple[str, ...] = ()


def token_count(text: str, tokenizer: Tokenizer) -> int:
    return len(tokenizer.encode(text))


def split_text_to_budget(
    text: str,
    *,
    tokenizer: Tokenizer,
    max_tokens: int,
) -> list[str]:
    """Prefer sentence boundaries; only slice raw tokens for an oversized sentence."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if token_count(text, tokenizer) <= max_tokens:
        return [text]

    sentences = [part.strip() for part in SENTENCE_BREAK_RE.split(text) if part.strip()]
    if not sentences:
        return split_raw_tokens(text, tokenizer=tokenizer, max_tokens=max_tokens)

    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for sentence in sentences:
        sentence_tokens = token_count(sentence, tokenizer)
        if sentence_tokens > max_tokens:
            if current:
                pieces.append("\n".join(current))
                current, current_tokens = [], 0
            pieces.extend(split_raw_tokens(sentence, tokenizer=tokenizer, max_tokens=max_tokens))
            continue

        separator_tokens = token_count("\n", tokenizer) if current else 0
        if current and current_tokens + separator_tokens + sentence_tokens > max_tokens:
            pieces.append("\n".join(current))
            current, current_tokens = [sentence], sentence_tokens
        else:
            current.append(sentence)
            current_tokens += separator_tokens + sentence_tokens

    if current:
        pieces.append("\n".join(current))
    return pieces


def split_raw_tokens(text: str, *, tokenizer: Tokenizer, max_tokens: int) -> list[str]:
    tokens = tokenizer.encode(text)
    return [
        tokenizer.decode(tokens[start : start + max_tokens]).strip()
        for start in range(0, len(tokens), max_tokens)
        if tokens[start : start + max_tokens]
    ]


def element_to_text_units(
    element: ParsedElement,
    *,
    tokenizer: Tokenizer,
    max_tokens: int,
) -> list[TokenTextUnit]:
    return [
        TokenTextUnit(
            text=piece,
            line_start=element.line_start,
            line_end=element.line_end,
            image_urls=element.image_urls,
        )
        for piece in split_text_to_budget(
            element.text,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
        )
    ]


def overlap_tail(
    units: list[TokenTextUnit], *, tokenizer: Tokenizer, overlap_tokens: int
) -> list[TokenTextUnit]:
    if overlap_tokens <= 0 or not units:
        return []

    selected: list[TokenTextUnit] = []
    used = 0
    for unit in reversed(units):
        count = token_count(unit.text, tokenizer)
        separator = token_count("\n\n", tokenizer) if selected else 0
        if used + separator + count <= overlap_tokens:
            selected.append(unit)
            used += separator + count
            continue

        remaining = overlap_tokens - used - separator
        if remaining > 0:
            tail = tokenizer.decode(tokenizer.encode(unit.text)[-remaining:]).strip()
            if tail:
                selected.append(
                    TokenTextUnit(
                        text=tail,
                        line_start=unit.line_start,
                        line_end=unit.line_end,
                        image_urls=unit.image_urls,
                    )
                )
        break
    return list(reversed(selected))


def build_token_jsonl_chunks(
    sections: list[Section],
    *,
    tokenizer: Tokenizer,
    source: str = DEFAULT_SOURCE,
    max_tokens: int = 450,
    overlap_tokens: int = 60,
) -> list[IngestedChunk]:
    """Create ai17-compatible chunks while applying the token budget to text only."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be in [0, max_tokens)")

    chunks: list[IngestedChunk] = []
    for section in sections:
        type_counts = {"text": 0, "table": 0, "code": 0}
        pending: list[TokenTextUnit] = []

        def emit_text() -> None:
            nonlocal pending
            if not pending:
                return
            text = "\n\n".join(unit.text for unit in pending)
            image_urls = unique_preserve_order(
                url for unit in pending for url in unit.image_urls
            )
            chunks.append(
                make_chunk(
                    section=section,
                    source=source,
                    chunk_type="text",
                    index=type_counts["text"],
                    text=text,
                    line_start=pending[0].line_start,
                    line_end=pending[-1].line_end,
                    image_urls=image_urls,
                )
            )
            type_counts["text"] += 1
            pending = overlap_tail(
                pending,
                tokenizer=tokenizer,
                overlap_tokens=overlap_tokens,
            )

        for element in parse_section_elements(section):
            if element.kind != "text":
                emit_text()
                pending = []
                type_counts[element.kind] += 1
                chunks.append(
                    make_chunk(
                        section=section,
                        source=source,
                        chunk_type=element.kind,
                        index=type_counts[element.kind] - 1,
                        text=element.text,
                        line_start=element.line_start,
                        line_end=element.line_end,
                        table=element.table,
                        code_block=element.code_block,
                    )
                )
                continue

            for unit in element_to_text_units(
                element,
                tokenizer=tokenizer,
                max_tokens=max_tokens,
            ):
                candidate = "\n\n".join([*(item.text for item in pending), unit.text])
                if pending and token_count(candidate, tokenizer) > max_tokens:
                    emit_text()
                    candidate = "\n\n".join([*(item.text for item in pending), unit.text])
                    while pending and token_count(candidate, tokenizer) > max_tokens:
                        pending.pop(0)
                        candidate = "\n\n".join(
                            [*(item.text for item in pending), unit.text]
                        )
                pending.append(unit)

        emit_text()
    return chunks


def ingest_markdown_by_tokens(
    input_path: Path,
    *,
    tokenizer: Tokenizer,
    source: str = DEFAULT_SOURCE,
    max_tokens: int = 450,
    overlap_tokens: int = 60,
) -> list[IngestedChunk]:
    sections = split_sections(read_markdown_lines(input_path))
    return build_token_jsonl_chunks(
        sections,
        tokenizer=tokenizer,
        source=source,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ai17-compatible JSONL using token-budget text chunks."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--max-tokens", type=int, default=450)
    parser.add_argument("--overlap-tokens", type=int, default=60)
    parser.add_argument("--encoding", default=DEFAULT_ENCODING)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = TiktokenTokenizer(args.encoding)
    chunks = ingest_markdown_by_tokens(
        args.input,
        tokenizer=tokenizer,
        source=args.source,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
    )
    write_jsonl(chunks, args.output)
    print(
        json.dumps(
            {
                **summarize(chunks),
                "output": str(args.output),
                "tokenizer_encoding": tokenizer.encoding_name,
                "max_tokens": args.max_tokens,
                "overlap_tokens": args.overlap_tokens,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
