"""Extract page-level text from PDF documents."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF


@dataclass
class PageContent:
    """Text extracted from a single PDF page."""

    page_number: int
    text: str


@dataclass
class ExtractedDocument:
    """Full text extraction result for one PDF file."""

    document_name: str
    source_path: Path
    pages: list[PageContent]

    @property
    def full_text(self) -> str:
        return "\n\n".join(page.text for page in self.pages if page.text.strip())


def extract_pdf(path: Path) -> ExtractedDocument:
    """
    Extract text from every page of a PDF.

    Uses PyMuPDF because it is fast, reliable for IAEA-style PDFs,
    and preserves page boundaries for citation metadata.
    """
    document_name = path.name
    pages: list[PageContent] = []

    with fitz.open(path) as doc:
        for index, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            pages.append(PageContent(page_number=index, text=text))

    return ExtractedDocument(
        document_name=document_name,
        source_path=path,
        pages=pages,
    )


def extract_all_pdfs(documents_dir: Path) -> list[ExtractedDocument]:
    """Extract all PDF files from the configured documents directory."""
    pdf_paths = sorted(documents_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {documents_dir}")
    return [extract_pdf(path) for path in pdf_paths]
