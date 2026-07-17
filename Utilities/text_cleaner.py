"""Clean and normalize extracted PDF text before chunking."""

from __future__ import annotations

import re
import unicodedata


def normalize_unicode(text: str) -> str:
    """Normalize unicode characters to a consistent form."""
    return unicodedata.normalize("NFKC", text)


def fix_hyphenated_line_breaks(text: str) -> str:
    """Join words split across line breaks (e.g. 'verifi-\\ncation')."""
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def collapse_whitespace(text: str) -> str:
    """Replace repeated spaces/tabs while preserving paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_page_artifacts(text: str) -> str:
    """Remove common PDF artifacts such as isolated page numbers."""
    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if re.fullmatch(r"\d{1,4}", stripped):
            continue
        if re.fullmatch(r"-\s*\d+\s*-", stripped):
            continue
        lines.append(line)
    return "\n".join(lines)


def clean_text(text: str) -> str:
    """
    Apply the full cleaning pipeline to raw PDF text.

    Steps mirror the lab philosophy: normalize, repair broken words,
    remove noise, and keep paragraph structure intact.
    """
    if not text:
        return ""

    cleaned = normalize_unicode(text)
    cleaned = fix_hyphenated_line_breaks(cleaned)
    cleaned = remove_page_artifacts(cleaned)
    cleaned = collapse_whitespace(cleaned)
    return cleaned


def clean_page_text(text: str) -> str:
    """Clean a single page while keeping page-local structure."""
    return clean_text(text)
