from __future__ import annotations

import threading
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModelLoadError(RuntimeError):
    pass


class EmbeddingService:
    """
    Local embedding generation via SentenceTransformers.

    - Loads the model once (expensive)
    - Provides a threadsafe encode method (FastAPI runs concurrent requests)
    """

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._lock = threading.Lock()
        self._model: SentenceTransformer | None = None

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    try:
                        self._model = SentenceTransformer(self._model_name)
                    except Exception as e:
                        raise EmbeddingModelLoadError(
                            "Failed to load embedding model. "
                            "This usually means the model download failed (no internet/proxy) "
                            "or the local cache is corrupted. "
                            f"Model: {self._model_name}"
                        ) from e
        return self._model

    def embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        model = self._get_model()
        # normalize_embeddings=True gives cosine-friendly vectors; we can use dot-product in FAISS.
        vectors = model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        return vectors

    def embed_query(self, text: str) -> np.ndarray:
        v = self.embed_texts([text])
        return v[0]

