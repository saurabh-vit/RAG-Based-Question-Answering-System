from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from app.services.embedding_service import EmbeddingService
from app.services.vector_store import FaissVectorStore
from app.utils.chunker import chunk_text_tokenwise
from app.utils.text_extract import extract_text_from_file


log = logging.getLogger("rag.ingestion")


@dataclass(frozen=True)
class IngestionResult:
    document_id: str
    chunks: int


class IngestionService:
    def __init__(
        self,
        *,
        embeddings: EmbeddingService,
        vector_store: FaissVectorStore,
        chunk_size_tokens: int,
        chunk_overlap_tokens: int,
    ) -> None:
        self._embeddings = embeddings
        self._store = vector_store
        self._chunk_size = chunk_size_tokens
        self._overlap = chunk_overlap_tokens

    def ingest_document(self, *, document_id: str, file_path: str | Path) -> IngestionResult:
        """
        End-to-end ingestion:
          file -> text -> chunks -> embeddings -> FAISS + SQLite metadata
        """

        path = Path(file_path)
        try:
            text = extract_text_from_file(path)
            text = _normalize_text(text)

            if len(text.strip()) < 20:
                log.warning("Document text too small", extra={"document_id": document_id, "file": str(path)})
                return IngestionResult(document_id=document_id, chunks=0)

            chunks = chunk_text_tokenwise(
                text,
                document_id=document_id,
                chunk_size_tokens=self._chunk_size,
                overlap_tokens=self._overlap,
            )

            vectors = self._embeddings.embed_texts([c.text for c in chunks])

            added = self._store.add_embeddings(
                document_id=document_id,
                chunk_ids=[c.chunk_id for c in chunks],
                chunk_texts=[c.text for c in chunks],
                start_tokens=[c.start_token for c in chunks],
                end_tokens=[c.end_token for c in chunks],
                embeddings=vectors,
            )

            log.info("Ingestion completed", extra={"document_id": document_id, "chunks": added, "file": str(path)})
            return IngestionResult(document_id=document_id, chunks=added)
        except Exception as e:
            log.exception("Ingestion failed", extra={"document_id": document_id, "file": str(path), "error": str(e)})
            return IngestionResult(document_id=document_id, chunks=0)


def _normalize_text(t: str) -> str:
    # Keep it conservative: normalize whitespace without losing structure.
    return "\n".join(line.rstrip() for line in t.replace("\r\n", "\n").replace("\r", "\n").split("\n"))

