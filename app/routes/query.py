from __future__ import annotations

import hashlib
import logging
import os

from fastapi import APIRouter, HTTPException, Request

from app.models.query import AskRequest, AskResponse, SourceChunk
from app.services.container import Container
from app.services.embedding_service import EmbeddingModelLoadError
from app.services.llm_service import build_context
from app.utils.highlight import highlight_terms
from app.utils.limiter import limiter
from app.utils.metrics import Timer
from app.utils.settings import get_settings


log = logging.getLogger("rag.routes.query")
router = APIRouter(tags=["query"])
settings = get_settings()


def _get_container(request: Request) -> Container:
    return request.app.state.container


@router.post("/ask", response_model=AskResponse)
@limiter.limit(settings.rate_limit_ask)
async def ask(request: Request, payload: AskRequest) -> AskResponse:
    container = _get_container(request)
    app_settings = container.settings

    timer = Timer.start_new()

    top_k = payload.top_k or app_settings.top_k
    top_k = max(3, top_k)  # retrieval must use at least 3 chunks
    doc_ids = payload.document_ids

    cache_key = _cache_key(question=payload.question, document_ids=doc_ids, top_k=top_k)
    cached = container.query_cache.get(cache_key)
    if cached is not None:
        resp: AskResponse = cached
        return AskResponse(**resp.model_dump(), cached=True, latency_ms=timer.elapsed_ms())

    try:
        qvec = container.embedding_service.embed_query(payload.question)
    except EmbeddingModelLoadError as e:
        # Service should not crash if the local model isn't available (e.g. offline).
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        # Catch-all to avoid 500s on transient HF cache/download/torch issues.
        raise HTTPException(status_code=503, detail=f"Embedding service failed: {e}")
    retrieved = container.vector_store.search(qvec, top_k=top_k, document_ids=doc_ids)

    log.info(
        "retrieval_summary",
        extra={
            "retrieved_chunks": len(retrieved),
            "top_k": top_k,
            "filtered_docs": len(doc_ids) if doc_ids else None,
        },
    )

    # Log similarity scores for observability (mandatory requirement).
    for r in retrieved:
        log.info(
            "retrieved_chunk",
            extra={"document_id": r.document_id, "chunk_id": r.chunk_id, "score": r.score},
        )

    if not retrieved:
        answer = "Not in document."
        resp = AskResponse(answer=answer, sources=[], cached=False, latency_ms=timer.elapsed_ms())
        container.query_cache.set(cache_key, resp)
        return resp

    context = build_context([r.text for r in retrieved])
    log.info("context_ready", extra={"context_length": len(context)})

    answer = container.llm_service.answer(question=payload.question, context=context).strip()

    if answer.lower().startswith("error generating answer:"):
        log.error("gemini_error", extra={"error": answer})
        raise HTTPException(status_code=503, detail=answer)
    if answer.lower().startswith("missing google ai studio key"):
        raise HTTPException(status_code=503, detail=answer)
    if not answer:
        answer = "Not in document."

    sources = [
        SourceChunk(
            document_id=r.document_id,
            chunk_id=r.chunk_id,
            score=r.score,
            text=r.text,
            highlighted_text=highlight_terms(r.text, payload.question),
        )
        for r in retrieved
    ]

    # Enforce strict grounding requirement
    if "not in document" in answer.lower() and answer.strip().lower() != "not in document.":
        # Normalize to the exact phrase (per requirements)
        answer = "Not in document."

    resp = AskResponse(answer=answer, sources=sources, cached=False, latency_ms=timer.elapsed_ms())
    container.query_cache.set(cache_key, resp)
    return resp


def _cache_key(*, question: str, document_ids: list[str] | None, top_k: int) -> str:
    h = hashlib.sha256()
    h.update(question.strip().encode("utf-8"))
    h.update(str(top_k).encode("utf-8"))
    if document_ids:
        for d in sorted(document_ids):
            h.update(d.encode("utf-8"))
    return h.hexdigest()

