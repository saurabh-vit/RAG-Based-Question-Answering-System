from __future__ import annotations

from app.services.embedding_service import EmbeddingService
from app.services.ingestion_service import IngestionService
from app.services.llm_service import LLMService
from app.services.query_cache import QueryCache
from app.services.vector_store import FaissVectorStore
from app.utils.settings import Settings


class Container:
    """
    Minimal dependency container (keeps wiring centralized and testable).
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

        self.vector_store = FaissVectorStore(settings.vector_store_dir)
        self.embedding_service = EmbeddingService(settings.embedding_model_name)
        self.ingestion_service = IngestionService(
            embeddings=self.embedding_service,
            vector_store=self.vector_store,
            chunk_size_tokens=settings.chunk_size_tokens,
            chunk_overlap_tokens=settings.chunk_overlap_tokens,
        )
        self.llm_service = LLMService()
        self.query_cache = QueryCache(ttl_seconds=settings.cache_ttl_seconds)

