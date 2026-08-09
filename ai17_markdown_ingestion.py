from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal


ChunkType = Literal["text", "table", "code"]

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = PROJECT_ROOT / "八股文.md"
DEFAULT_SOURCE = "八股文.md"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "bagu_chunks.jsonl"

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
FENCE_RE = re.compile(r"^\s*```(.*)\s*$")
IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
HEADING_NUMBER_RE = re.compile(r"^[一二三四五六七八九十百零〇两]+\s*、\s*")
MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[([^\]]+)\]\(([^)]+)\)")
INLINE_CODE_RE = re.compile(r"`([^`]+)`")
TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")


@dataclass(frozen=True)
class MarkdownLine:
    number: int
    text: str


@dataclass(frozen=True)
class TableData:
    columns: list[str]
    rows: list[dict[str, str]]
    markdown: str


@dataclass(frozen=True)
class CodeBlock:
    language: str
    content: str
    line_start: int
    line_end: int


@dataclass(frozen=True)
class ParsedElement:
    kind: ChunkType
    text: str
    line_start: int
    line_end: int
    table: TableData | None = None
    code_block: CodeBlock | None = None
    image_urls: tuple[str, ...] = ()


@dataclass(frozen=True)
class Section:
    index: int
    title: str
    heading_path: list[str]
    line_start: int
    line_end: int
    lines: list[MarkdownLine]


@dataclass(frozen=True)
class IngestedChunk:
    id: str
    source: str
    title: str
    heading_path: list[str]
    chunk_type: ChunkType
    text: str
    tables: list[dict[str, Any]]
    code_blocks: list[dict[str, Any]]
    image_urls: list[str]
    metadata: dict[str, Any]


def read_markdown_lines(path: Path) -> list[MarkdownLine]:
    return [
        MarkdownLine(number=index, text=line.rstrip("\n"))
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
    ]


def split_sections(lines: list[MarkdownLine]) -> list[Section]:
    sections: list[Section] = []
    heading_path: list[str] = []
    current_title = "(root)"
    current_heading_path: list[str] = []
    current_start = 1
    current_lines: list[MarkdownLine] = []
    section_index = 0
    in_code = False

    def flush(end_line: int) -> None:
        nonlocal section_index, current_lines
        if not current_lines and current_title == "(root)":
            return

        section_index += 1
        sections.append(
            Section(
                index=section_index,
                title=current_title,
                heading_path=current_heading_path.copy(),
                line_start=current_start,
                line_end=end_line,
                lines=current_lines,
            )
        )
        current_lines = []

    for line in lines:
        stripped = line.text.strip()
        if FENCE_RE.match(stripped):
            in_code = not in_code

        heading_match = HEADING_RE.match(stripped) if not in_code else None
        if heading_match:
            flush(line.number - 1)
            level = len(heading_match.group(1))
            raw_title = heading_match.group(2).strip()
            heading_path = heading_path[: level - 1] + [raw_title]
            current_title = clean_heading_title(raw_title)
            current_heading_path = heading_path.copy()
            current_start = line.number
            current_lines = []
            continue

        current_lines.append(line)

    if lines:
        flush(lines[-1].number)
    return sections


def parse_section_elements(section: Section) -> list[ParsedElement]:
    elements: list[ParsedElement] = []
    text_buffer: list[MarkdownLine] = []
    i = 0

    def flush_text() -> None:
        nonlocal text_buffer
        paragraphs = split_paragraphs(text_buffer)
        for paragraph in paragraphs:
            raw_body = "\n".join(line.text.strip() for line in paragraph).strip()
            image_urls = tuple(extract_image_urls(raw_body))
            body = clean_markdown_text(raw_body)
            if not body and not image_urls:
                continue
            elements.append(
                ParsedElement(
                    kind="text",
                    text=body,
                    line_start=paragraph[0].number,
                    line_end=paragraph[-1].number,
                    image_urls=image_urls,
                )
            )
        text_buffer = []

    while i < len(section.lines):
        line = section.lines[i]
        stripped = line.text.strip()

        fence_match = FENCE_RE.match(stripped)
        if fence_match:
            flush_text()
            language = fence_match.group(1).strip()
            code_lines: list[MarkdownLine] = []
            start_line = line.number
            i += 1
            while i < len(section.lines):
                candidate = section.lines[i]
                if FENCE_RE.match(candidate.text.strip()):
                    break
                code_lines.append(candidate)
                i += 1

            end_line = section.lines[i].number if i < len(section.lines) else (
                code_lines[-1].number if code_lines else start_line
            )
            content = "\n".join(item.text for item in code_lines).rstrip()
            code_block = CodeBlock(
                language=language,
                content=content,
                line_start=start_line,
                line_end=end_line,
            )
            elements.append(
                ParsedElement(
                    kind="code",
                    text=content,
                    line_start=start_line,
                    line_end=end_line,
                    code_block=code_block,
                )
            )
            i += 1
            continue

        if is_table_start(section.lines, i):
            flush_text()
            table_lines = collect_table_lines(section.lines, i)
            table = parse_table(table_lines)
            elements.append(
                ParsedElement(
                    kind="table",
                    text=flatten_table(table),
                    line_start=table_lines[0].number,
                    line_end=table_lines[-1].number,
                    table=table,
                )
            )
            i += len(table_lines)
            continue

        text_buffer.append(line)
        i += 1

    flush_text()
    return elements


def split_paragraphs(lines: list[MarkdownLine]) -> list[list[MarkdownLine]]:
    paragraphs: list[list[MarkdownLine]] = []
    current: list[MarkdownLine] = []
    for line in lines:
        if not line.text.strip():
            if current:
                paragraphs.append(current)
                current = []
            continue
        current.append(line)
    if current:
        paragraphs.append(current)
    return paragraphs


def is_table_start(lines: list[MarkdownLine], index: int) -> bool:
    if index + 1 >= len(lines):
        return False
    if not TABLE_LINE_RE.match(lines[index].text):
        return False
    if not TABLE_LINE_RE.match(lines[index + 1].text):
        return False
    return is_table_separator(split_table_cells(lines[index + 1].text))


def collect_table_lines(lines: list[MarkdownLine], start: int) -> list[MarkdownLine]:
    collected: list[MarkdownLine] = []
    i = start
    while i < len(lines) and TABLE_LINE_RE.match(lines[i].text):
        collected.append(lines[i])
        i += 1
    return collected


def parse_table(lines: list[MarkdownLine]) -> TableData:
    markdown = "\n".join(clean_markdown_text(line.text) for line in lines)
    raw_rows = [
        [clean_markdown_text(cell) for cell in split_table_cells(line.text)]
        for line in lines
    ]
    columns = raw_rows[0] if raw_rows else []
    data_rows = raw_rows[2:] if len(raw_rows) >= 2 and is_table_separator(raw_rows[1]) else raw_rows[1:]

    rows: list[dict[str, str]] = []
    for raw_row in data_rows:
        row: dict[str, str] = {}
        for index, column in enumerate(columns):
            column_name = normalize_table_column(column, index)
            row[column_name] = raw_row[index] if index < len(raw_row) else ""
        rows.append(row)

    return TableData(columns=columns, rows=rows, markdown=markdown)


def normalize_table_column(column: str, index: int) -> str:
    if column:
        return column
    if index == 0:
        return "对比项"
    return f"column_{index + 1}"


def split_table_cells(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def is_table_separator(cells: list[str]) -> bool:
    if not cells:
        return False
    for cell in cells:
        normalized = cell.replace(" ", "")
        if not re.fullmatch(r":?-{3,}:?", normalized):
            return False
    return True


def flatten_table(table: TableData) -> str:
    flattened_rows: list[str] = []
    for row in table.rows:
        parts = [f"{column}={value}" for column, value in row.items() if value]
        if parts:
            flattened_rows.append("; ".join(parts))
    return "\n".join(flattened_rows)


def extract_image_urls(text: str) -> list[str]:
    return [match.group(1).strip() for match in IMAGE_RE.finditer(text)]


def clean_heading_title(title: str) -> str:
    return HEADING_NUMBER_RE.sub("", title).strip()


def clean_markdown_text(text: str) -> str:
    text = IMAGE_RE.sub("", text)
    text = MARKDOWN_LINK_RE.sub(r"\1", text)
    text = INLINE_CODE_RE.sub(r"\1", text)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"__(.*?)__", r"\1", text, flags=re.DOTALL)
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def build_jsonl_chunks(
    sections: list[Section],
    source: str = DEFAULT_SOURCE,
    max_text_chars: int = 1200,
) -> list[IngestedChunk]:
    chunks: list[IngestedChunk] = []
    for section in sections:
        elements = parse_section_elements(section)
        text_group: list[ParsedElement] = []
        text_group_chars = 0
        type_counts: dict[ChunkType, int] = {"text": 0, "table": 0, "code": 0}

        def emit_text_group() -> None:
            nonlocal text_group, text_group_chars
            if not text_group:
                return

            type_counts["text"] += 1
            text = "\n\n".join(element.text for element in text_group if element.text)
            image_urls = unique_preserve_order(
                url for element in text_group for url in element.image_urls
            )
            chunks.append(
                make_chunk(
                    section=section,
                    source=source,
                    chunk_type="text",
                    index=type_counts["text"] - 1,
                    text=text,
                    line_start=text_group[0].line_start,
                    line_end=text_group[-1].line_end,
                    image_urls=image_urls,
                )
            )
            text_group = []
            text_group_chars = 0

        for element in elements:
            if element.kind != "text":
                emit_text_group()
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

            candidate_chars = text_group_chars + len(element.text) + (2 if text_group else 0)
            if text_group and candidate_chars > max_text_chars:
                emit_text_group()

            text_group.append(element)
            text_group_chars += len(element.text) + (2 if len(text_group) > 1 else 0)

        emit_text_group()

    return chunks


def make_chunk(
    section: Section,
    source: str,
    chunk_type: ChunkType,
    index: int,
    text: str,
    line_start: int,
    line_end: int,
    table: TableData | None = None,
    code_block: CodeBlock | None = None,
    image_urls: list[str] | None = None,
) -> IngestedChunk:
    chunk_id = f"bagu_{section.index:04d}_{chunk_type}_{index}"
    return IngestedChunk(
        id=chunk_id,
        source=source,
        title=section.title,
        heading_path=section.heading_path,
        chunk_type=chunk_type,
        text=text,
        tables=[asdict(table)] if table is not None else [],
        code_blocks=[asdict(code_block)] if code_block is not None else [],
        image_urls=image_urls or [],
        metadata={
            "section_index": section.index,
            "section_line_start": section.line_start,
            "section_line_end": section.line_end,
            "line_start": line_start,
            "line_end": line_end,
            "char_count": len(text),
        },
    )


def unique_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def write_jsonl(chunks: list[IngestedChunk], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as file:
        for chunk in chunks:
            file.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def summarize(chunks: list[IngestedChunk]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for chunk in chunks:
        counts[chunk.chunk_type] = counts.get(chunk.chunk_type, 0) + 1

    return {
        "total_chunks": len(chunks),
        "chunk_type_counts": counts,
        "sections": len({chunk.metadata["section_index"] for chunk in chunks}),
        "image_url_chunks": sum(1 for chunk in chunks if chunk.image_urls),
    }


def ingest_markdown(
    input_path: Path,
    source: str = DEFAULT_SOURCE,
    max_text_chars: int = 1200,
) -> list[IngestedChunk]:
    lines = read_markdown_lines(input_path)
    sections = split_sections(lines)
    return build_jsonl_chunks(sections, source=source, max_text_chars=max_text_chars)


def demo() -> None:
    chunks = ingest_markdown(DEFAULT_INPUT_PATH)
    summary = summarize(chunks)

    assert summary["total_chunks"] > 0
    assert summary["chunk_type_counts"].get("text", 0) > 0
    assert summary["chunk_type_counts"].get("table", 0) > 0
    assert summary["chunk_type_counts"].get("code", 0) > 0
    assert summary["image_url_chunks"] > 0
    assert all(chunk.metadata["line_start"] <= chunk.metadata["line_end"] for chunk in chunks)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for chunk in chunks[:3]:
        print(json.dumps(asdict(chunk), ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest semi-structured Markdown into JSONL chunks.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--source", type=str, default=DEFAULT_SOURCE)
    parser.add_argument("--max-text-chars", type=int, default=1200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunks = ingest_markdown(
        input_path=args.input,
        source=args.source,
        max_text_chars=args.max_text_chars,
    )

    if args.output is not None:
        write_jsonl(chunks, args.output)

    print(json.dumps(summarize(chunks), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
