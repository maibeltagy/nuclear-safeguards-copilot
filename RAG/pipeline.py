"""End-to-end RAG orchestration for Nuclear Safeguards Copilot."""

from __future__ import annotations

from dataclasses import dataclass

from Configuration.settings import settings
from Embeddings.embedder import Embedder
from RAG.context_builder import ContextPackage, build_context_package
from RAG.llm_client import BaseLLMClient, create_llm_client
from RAG.prompt_builder import NOT_FOUND_MESSAGE, build_grounded_prompt
from RAG.retriever.base import RetrievalResult
from RAG.retriever.semantic_retriever import SemanticRetriever
from Vector_Database.faiss_store import FaissStore


@dataclass
class SourceCitation:
    document_name: str
    section: str
    page_number: int
    page_end: int
    chunk_id: str
    similarity_score: float
    excerpt: str


@dataclass
class ChatResponse:
    question: str
    answer: str
    sources: list[SourceCitation]
    context_used_words: int


class RAGPipeline:
    """
    Full RAG pipeline:
    retrieve -> context package -> grounded prompt -> LLM -> answer + citations
    """

    def __init__(
        self,
        store: FaissStore | None = None,
        embedder: Embedder | None = None,
        retriever: SemanticRetriever | None = None,
        llm_client: BaseLLMClient | None = None,
    ) -> None:
        self.store = store or FaissStore.load(
            settings.paths.faiss_index_file,
            settings.paths.chunks_metadata_file,
        )
        self.embedder = embedder or Embedder()
        self.retriever = retriever or SemanticRetriever(self.store, self.embedder)
        self.llm_client = llm_client or create_llm_client()

    def retrieve(self, question: str) -> list[RetrievalResult]:
        return self.retriever.retrieve(question, top_k=settings.retrieval.top_k)

    def build_context(self, question: str, candidates: list[RetrievalResult]) -> ContextPackage:
        return build_context_package(question, candidates)

    def answer(self, question: str) -> ChatResponse:
        question = question.strip()
        if not question:
            return ChatResponse(
                question=question,
                answer="Please enter a question about IAEA nuclear safeguards documents.",
                sources=[],
                context_used_words=0,
            )

        candidates = self.retrieve(question)
        context_package = self.build_context(question, candidates)

        if not context_package.selected_results:
            return ChatResponse(
                question=question,
                answer=NOT_FOUND_MESSAGE,
                sources=[],
                context_used_words=0,
            )

        prompt = build_grounded_prompt(question, context_package.context_text)
        llm_answer = self.llm_client.generate(prompt).strip()

        sources = [
            SourceCitation(
                document_name=result.chunk.document_name,
                section=result.chunk.section,
                page_number=result.chunk.page_number,
                page_end=result.chunk.page_end,
                chunk_id=result.chunk.chunk_id,
                similarity_score=round(result.score, 4),
                excerpt=result.chunk.chunk_text[:280] + (
                    "..." if len(result.chunk.chunk_text) > 280 else ""
                ),
            )
            for result in context_package.selected_results
        ]

        return ChatResponse(
            question=question,
            answer=llm_answer or NOT_FOUND_MESSAGE,
            sources=sources,
            context_used_words=context_package.used_words,
        )
