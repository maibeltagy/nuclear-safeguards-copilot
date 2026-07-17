"""Chat API route."""

from __future__ import annotations

import traceback

from functools import lru_cache

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from RAG.pipeline import RAGPipeline

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class SourceResponse(BaseModel):
    document_name: str
    section: str
    page_number: int
    page_end: int
    chunk_id: str
    similarity_score: float
    excerpt: str


class ChatResponseModel(BaseModel):
    question: str
    answer: str
    sources: list[SourceResponse]
    context_used_words: int


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    """Load the RAG pipeline once and reuse it across requests."""
    return RAGPipeline()


@router.post("/chat", response_model=ChatResponseModel)
def chat(request: ChatRequest) -> ChatResponseModel:
    """
    Single MVP endpoint:
    question -> retrieval -> context -> LLM -> grounded answer + citations
    """
    try:
        pipeline = get_pipeline()
        result = pipeline.answer(request.question.strip())
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"RAG pipeline error: {exc}",
        ) from exc

    return ChatResponseModel(
        question=result.question,
        answer=result.answer,
        sources=[SourceResponse(**source.__dict__) for source in result.sources],
        context_used_words=result.context_used_words,
    )
