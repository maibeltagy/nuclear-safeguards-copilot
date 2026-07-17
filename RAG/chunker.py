"""
Structure-aware manual chunker for IAEA safeguard PDFs.

Inspired by Lab 8/9 chunking philosophy but adapted for real PDF structure:
preserve paragraphs, headings, lists, tables, and glossary entries as atomic blocks.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from Configuration.settings import ChunkingSettings, settings
from Utilities.pdf_extractor import ExtractedDocument
from Utilities.text_cleaner import clean_page_text

STOPWORDS = {
    "about", "after", "also", "and", "are", "been", "being", "between",
    "both", "but", "can", "could", "each", "for", "from", "have", "has",
    "into", "its", "more", "most", "not", "other", "shall", "such", "than",
    "that", "the", "their", "them", "then", "there", "these", "they", "this",
    "those", "through", "under", "used", "using", "very", "were", "which",
    "with", "within", "would",
}


@dataclass
class TextBlock:
    """A semantic unit that must not be split during chunking."""

    block_type: str  # paragraph | heading | list | table | glossary
    text: str
    page_number: int
    section: str


@dataclass
class ChunkRecord:
    """One retrievable chunk with full metadata."""

    chunk_id: str
    document_name: str
    section: str
    page_number: int
    page_end: int
    keywords: list[str]
    chunk_position: int
    word_count: int
    chunk_text: str
    search_text: str

    def to_dict(self) -> dict:
        return asdict(self)


def word_count(text: str) -> int:
    return len(text.split())


def extract_keywords(text: str, limit: int = 8) -> list[str]:
    """Extract simple keyword candidates from chunk text."""
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", text.lower())
    freq: dict[str, int] = {}
    for token in tokens:
        if token in STOPWORDS or len(token) < 4:
            continue
        freq[token] = freq.get(token, 0) + 1

    ranked = sorted(freq.items(), key=lambda item: (-item[1], item[0]))
    return [word for word, _ in ranked[:limit]]


def is_heading_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        return False
    if re.match(r"^\d+(\.\d+)*[\.\)]?\s+[A-Z]", stripped):
        return True
    if stripped.isupper() and len(stripped.split()) <= 12:
        return True
    if re.match(r"^(Chapter|Section|Appendix|Part)\s+\d+", stripped, re.I):
        return True
    if stripped.endswith(":") and len(stripped.split()) <= 10:
        return True
    return False


def is_list_line(line: str) -> bool:
    stripped = line.strip()
    return bool(
        re.match(r"^(\u2022|\u2023|\u25E6|\u2043|\u2219|[-*•])\s+", stripped)
        or re.match(r"^\(?[a-zA-Z0-9]\)?[\.\)]\s+", stripped)
        or re.match(r"^\d+[\.\)]\s+", stripped)
    )


def is_table_line(line: str) -> bool:
    stripped = line.strip()
    if "|" in stripped and stripped.count("|") >= 2:
        return True
    if re.search(r"\t", stripped):
        return True
    if len(stripped.split()) >= 4 and re.search(r"\s{2,}", stripped):
        return True
    return False


def is_glossary_line(line: str) -> bool:
    stripped = line.strip()
    if re.match(r"^[A-Z][A-Za-z0-9 /-]{1,60}\s*[—–-]\s+\w", stripped):
        return True
    if re.match(r"^[A-Z][A-Za-z0-9 /-]{1,60}:\s+\w", stripped):
        return True
    return False


def split_into_lines(text: str) -> list[str]:
    return [line.rstrip() for line in text.split("\n")]


def parse_page_blocks(page_text: str, page_number: int, current_section: str) -> tuple[list[TextBlock], str]:
    """
    Parse one page into semantic blocks without breaking structure.

    Returns blocks and the updated section title from headings found on the page.
    """
    blocks: list[TextBlock] = []
    lines = split_into_lines(page_text)

    paragraph_buffer: list[str] = []
    list_buffer: list[str] = []
    table_buffer: list[str] = []
    glossary_buffer: list[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph_buffer
        if paragraph_buffer:
            text = " ".join(part.strip() for part in paragraph_buffer if part.strip())
            if text:
                blocks.append(
                    TextBlock("paragraph", text, page_number, current_section)
                )
            paragraph_buffer = []

    def flush_list() -> None:
        nonlocal list_buffer
        if list_buffer:
            blocks.append(
                TextBlock("list", "\n".join(list_buffer), page_number, current_section)
            )
            list_buffer = []

    def flush_table() -> None:
        nonlocal table_buffer
        if table_buffer:
            blocks.append(
                TextBlock("table", "\n".join(table_buffer), page_number, current_section)
            )
            table_buffer = []

    def flush_glossary() -> None:
        nonlocal glossary_buffer
        if glossary_buffer:
            blocks.append(
                TextBlock(
                    "glossary",
                    "\n".join(glossary_buffer),
                    page_number,
                    current_section,
                )
            )
            glossary_buffer = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            flush_list()
            flush_table()
            flush_glossary()
            continue

        if is_heading_line(stripped):
            flush_paragraph()
            flush_list()
            flush_table()
            flush_glossary()
            current_section = stripped.rstrip(":")
            blocks.append(
                TextBlock("heading", stripped, page_number, current_section)
            )
            continue

        if is_glossary_line(stripped):
            flush_paragraph()
            flush_list()
            flush_table()
            glossary_buffer.append(stripped)
            continue

        if is_table_line(stripped):
            flush_paragraph()
            flush_list()
            flush_glossary()
            table_buffer.append(stripped)
            continue

        if is_list_line(stripped):
            flush_paragraph()
            flush_table()
            flush_glossary()
            list_buffer.append(stripped)
            continue

        flush_list()
        flush_table()
        flush_glossary()
        paragraph_buffer.append(stripped)

    flush_paragraph()
    flush_list()
    flush_table()
    flush_glossary()
    return blocks, current_section


def split_long_block(block: TextBlock, max_words: int) -> list[TextBlock]:
    """
    Split an oversized block at sentence boundaries as a last resort.

    Headings, lists, tables, and glossary entries are kept intact even if long.
    """
    if block.block_type in {"heading", "list", "table", "glossary"}:
        return [block]

    if word_count(block.text) <= max_words:
        return [block]

    sentences = re.split(r"(?<=[.!?])\s+", block.text)
    parts: list[TextBlock] = []
    current: list[str] = []
    current_words = 0

    for sentence in sentences:
        sentence_words = word_count(sentence)
        if current and current_words + sentence_words > max_words:
            parts.append(
                TextBlock(
                    block.block_type,
                    " ".join(current),
                    block.page_number,
                    block.section,
                )
            )
            current = [sentence]
            current_words = sentence_words
        else:
            current.append(sentence)
            current_words += sentence_words

    if current:
        parts.append(
            TextBlock(
                block.block_type,
                " ".join(current),
                block.page_number,
                block.section,
            )
        )
    return parts


def build_search_text(document_name: str, section: str, keywords: list[str], chunk_text: str) -> str:
    """Prepend document context to chunk text for better retrieval (Lab 8 pattern)."""
    keyword_text = " ".join(keywords)
    return f"{document_name} {section} {keyword_text} {chunk_text}".strip()


def merge_blocks_into_chunks(
    blocks: Iterable[TextBlock],
    document_name: str,
    chunking: ChunkingSettings,
) -> list[ChunkRecord]:
    """
    Merge semantic blocks into bounded chunks without splitting atomic units.

    Uses overlap at sentence level when flushing a chunk, similar to Lab overlap idea
    but respecting document structure instead of blind word windows.
    """
    chunks: list[ChunkRecord] = []
    current_parts: list[TextBlock] = []
    current_words = 0
    position = 0

    def flush_chunk(force: bool = False) -> None:
        nonlocal current_parts, current_words, position
        if not current_parts:
            return

        chunk_words = sum(word_count(part.text) for part in current_parts)
        if not force and chunk_words < chunking.min_words and chunks:
            return

        chunk_text = "\n\n".join(part.text for part in current_parts)
        first_page = current_parts[0].page_number
        last_page = current_parts[-1].page_number
        section = next(
            (part.section for part in reversed(current_parts) if part.section),
            "General",
        )
        keywords = extract_keywords(chunk_text)
        chunk_id = f"{document_stem(document_name)}_chunk_{position}"
        record = ChunkRecord(
            chunk_id=chunk_id,
            document_name=document_name,
            section=section,
            page_number=first_page,
            page_end=last_page,
            keywords=keywords,
            chunk_position=position,
            word_count=word_count(chunk_text),
            chunk_text=chunk_text,
            search_text=build_search_text(document_name, section, keywords, chunk_text),
        )
        chunks.append(record)
        position += 1

        if chunking.overlap_sentences > 0 and current_parts:
            tail = current_parts[-1].text
            sentences = re.split(r"(?<=[.!?])\s+", tail)
            overlap = " ".join(sentences[-chunking.overlap_sentences :])
            if overlap.strip():
                current_parts = [
                    TextBlock(
                        current_parts[-1].block_type,
                        overlap,
                        current_parts[-1].page_number,
                        current_parts[-1].section,
                    )
                ]
                current_words = word_count(overlap)
                return

        current_parts = []
        current_words = 0

    for block in blocks:
        for unit in split_long_block(block, chunking.max_words):
            unit_words = word_count(unit.text)
            if unit_words > chunking.max_words and unit.block_type == "paragraph":
                # Extremely long paragraph already sentence-split; add directly.
                if current_parts:
                    flush_chunk(force=True)
                current_parts = [unit]
                current_words = unit_words
                flush_chunk(force=True)
                continue

            if current_words + unit_words > chunking.max_words and current_parts:
                flush_chunk(force=True)

            current_parts.append(unit)
            current_words += unit_words

    flush_chunk(force=True)
    return chunks


def document_stem(filename: str) -> str:
    """Derive a stable chunk-id prefix from a PDF filename."""
    return re.sub(r"[^A-Za-z0-9]+", "_", Path(filename).stem).strip("_").lower()


def chunk_document(
    document: ExtractedDocument,
    chunking: ChunkingSettings | None = None,
) -> list[ChunkRecord]:
    """Chunk one extracted PDF document."""
    chunking = chunking or settings.chunking
    all_blocks: list[TextBlock] = []
    current_section = "Introduction"

    for page in document.pages:
        cleaned = clean_page_text(page.text)
        if not cleaned:
            continue
        page_blocks, current_section = parse_page_blocks(
            cleaned, page.page_number, current_section
        )
        all_blocks.extend(page_blocks)

    return merge_blocks_into_chunks(all_blocks, document.document_name, chunking)


def chunk_documents(documents: list[ExtractedDocument]) -> list[ChunkRecord]:
    """Chunk multiple documents and assign globally unique chunk ids."""
    all_chunks: list[ChunkRecord] = []
    for doc_index, document in enumerate(documents):
        doc_chunks = chunk_document(document)
        for chunk in doc_chunks:
            chunk.chunk_id = f"doc{doc_index}_{chunk.chunk_id}"
        all_chunks.extend(doc_chunks)
    return all_chunks
