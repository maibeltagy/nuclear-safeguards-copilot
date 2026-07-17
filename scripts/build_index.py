"""
Build the FAISS index from all PDFs in Data/documents.

Run once after adding new PDFs:
    python scripts/build_index.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Configuration.settings import settings
from Embeddings.embedder import Embedder
from RAG.chunker import chunk_documents
from Utilities.pdf_extractor import extract_all_pdfs
from Vector_Database.faiss_store import FaissStore


def main() -> None:
    print("=" * 60)
    print("Nuclear Safeguards Copilot — Index Builder")
    print("=" * 60)

    documents_dir = settings.paths.documents_dir
    print(f"Documents directory: {documents_dir}")

    documents = extract_all_pdfs(documents_dir)
    print(f"Extracted {len(documents)} PDF(s)")

    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks")

    embedder = Embedder()
    texts = [chunk.search_text for chunk in chunks]
    print(f"Embedding with model: {settings.embedding.model_name}")
    embeddings = embedder.encode(texts)
    print(f"Embedding matrix shape: {embeddings.shape}")

    store = FaissStore(dimension=embedder.dimension)
    store.add(embeddings, chunks)

    settings.paths.index_dir.mkdir(parents=True, exist_ok=True)
    store.save(settings.paths.faiss_index_file, settings.paths.chunks_metadata_file)

    print(f"Saved FAISS index: {settings.paths.faiss_index_file}")
    print(f"Saved metadata:   {settings.paths.chunks_metadata_file}")
    print("Done.")


if __name__ == "__main__":
    main()
