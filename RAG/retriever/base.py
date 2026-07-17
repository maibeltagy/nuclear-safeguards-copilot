"""Retriever interfaces and shared result models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from RAG.chunker import ChunkRecord


@dataclass
class RetrievalResult:
    """One retrieved chunk with similarity score."""

    chunk: ChunkRecord
    score: float


class BaseRetriever(ABC):
    """
    Abstract retriever interface.

    SemanticRetriever implements this now; BM25Retriever or HybridRetriever
    can be added later without changing the RAG pipeline.
    """

    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        raise NotImplementedError
