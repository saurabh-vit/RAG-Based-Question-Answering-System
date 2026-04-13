from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    document_ids: list[str] | None = Field(
        default=None,
        description="Optional filter to restrict retrieval to specific documents (multi-doc support).",
    )
    top_k: int | None = Field(default=None, ge=1, le=10, description="Override default retrieval K (1..10).")


class SourceChunk(BaseModel):
    document_id: str
    chunk_id: str
    score: float = Field(..., description="Similarity score (higher is more similar).")
    text: str
    highlighted_text: str | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    cached: bool = False
    latency_ms: float

