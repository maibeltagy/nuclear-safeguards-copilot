"""
Build a grounded context package from retrieved chunks.

Implements Lab 8 context-building ideas: deduplicate, rank, budget control,
and labeled source blocks for the LLM prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from Configuration.settings import RetrievalSettings, settings
from RAG.retriever.base import RetrievalResult


@dataclass
class ContextPackage:
    """Final evidence package passed to the prompt builder."""

    query: str
    context_text: str
    selected_results: list[RetrievalResult]
    used_words: int


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def build_context_package(
    query: str,
    candidates: list[RetrievalResult],
    config: RetrievalSettings | None = None,
) -> ContextPackage:
    """
    Turn raw retrieval hits into a deduplicated, budgeted context package.

    Steps (from Lab 8):
    1. Sort by similarity score
    2. Drop low-score candidates
    3. Deduplicate near-identical chunks
    4. Limit chunks per document
    5. Respect word budget and max chunk count
    6. Build labeled source blocks with document + page metadata
    """
    config = config or settings.retrieval

    ranked = sorted(candidates, key=lambda item: item.score, reverse=True)
    max_score = ranked[0].score if ranked else 0.0

    selected: list[RetrievalResult] = []
    seen_texts: set[str] = set()
    per_document: dict[str, int] = {}
    used_words = 0

    for result in ranked:
        if result.score < config.min_absolute_score:
            continue
        if max_score > 0 and result.score < max_score * config.min_score_ratio:
            continue

        normalized = normalize_text(result.chunk.chunk_text)
        if normalized in seen_texts:
            continue

        doc_name = result.chunk.document_name
        if per_document.get(doc_name, 0) >= config.max_chunks_per_document:
            continue

        chunk_words = result.chunk.word_count
        if selected and used_words + chunk_words > config.word_budget:
            continue

        selected.append(result)
        seen_texts.add(normalized)
        per_document[doc_name] = per_document.get(doc_name, 0) + 1
        used_words += chunk_words

        if len(selected) >= config.max_context_chunks:
            break

    blocks: list[str] = []
    for index, result in enumerate(selected, start=1):
        chunk = result.chunk
        page_label = (
            f"p. {chunk.page_number}"
            if chunk.page_number == chunk.page_end
            else f"pp. {chunk.page_number}-{chunk.page_end}"
        )
        blocks.append(
            f"[Source {index}] Document: {chunk.document_name} | Section: {chunk.section} | "
            f"Page: {page_label} | Similarity: {result.score:.3f}\n"
            f"{chunk.chunk_text}"
        )

    return ContextPackage(
        query=query,
        context_text="\n\n".join(blocks),
        selected_results=selected,
        used_words=used_words,
    )
