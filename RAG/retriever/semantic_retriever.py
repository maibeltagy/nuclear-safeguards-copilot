"""Dense semantic retrieval using Sentence Transformers + FAISS."""

from __future__ import annotations

from Embeddings.embedder import Embedder
from RAG.retriever.base import BaseRetriever, RetrievalResult
from Vector_Database.faiss_store import FaissStore


class SemanticRetriever(BaseRetriever):
    """Embed the query and search the FAISS index for nearest chunks."""

    def __init__(self, store: FaissStore, embedder: Embedder | None = None) -> None:
        self.store = store
        self.embedder = embedder or Embedder()

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        query_vector = self.embedder.encode_query(query)
        hits = self.store.search(query_vector, top_k=top_k)
        return [RetrievalResult(chunk=chunk, score=score) for chunk, score in hits]
