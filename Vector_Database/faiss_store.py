"""FAISS vector store for chunk embeddings and metadata."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from RAG.chunker import ChunkRecord


class FaissStore:
    """
    Persist and query a flat inner-product FAISS index.

    Vectors are L2-normalized at embedding time, so inner product == cosine similarity.
    """

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: list[ChunkRecord] = []

    def add(self, embeddings: np.ndarray, chunks: list[ChunkRecord]) -> None:
        """Add embeddings and associated chunk metadata."""
        if embeddings.shape[0] != len(chunks):
            raise ValueError("Embeddings count must match chunk count")
        if embeddings.shape[1] != self.dimension:
            raise ValueError("Embedding dimension mismatch")

        self.index.add(embeddings.astype(np.float32))
        self.chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[ChunkRecord, float]]:
        """Return top-k chunks with similarity scores."""
        if self.index.ntotal == 0:
            return []

        query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        scores, indices = self.index.search(query, min(top_k, self.index.ntotal))

        results: list[tuple[ChunkRecord, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append((self.chunks[int(idx)], float(score)))
        return results

    def save(self, index_path: Path, metadata_path: Path) -> None:
        """Persist FAISS index and chunk metadata to disk."""
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))

        payload = [chunk.to_dict() for chunk in self.chunks]
        metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, index_path: Path, metadata_path: Path) -> "FaissStore":
        """Load a previously built index from disk."""
        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"Index not found. Run scripts/build_index.py first.\n"
                f"Expected: {index_path} and {metadata_path}"
            )

        index = faiss.read_index(str(index_path))
        dimension = index.d
        store = cls(dimension=dimension)
        store.index = index

        raw_chunks: list[dict[str, Any]] = json.loads(metadata_path.read_text(encoding="utf-8"))
        store.chunks = [ChunkRecord(**item) for item in raw_chunks]
        return store

    def __len__(self) -> int:
        return len(self.chunks)
