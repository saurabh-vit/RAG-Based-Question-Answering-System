from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from app.utils.files import ensure_dir


@dataclass(frozen=True)
class RetrievedChunk:
    document_id: str
    chunk_id: str
    score: float
    text: str


class FaissVectorStore:
    """
    FAISS index + SQLite metadata (production-ish local setup).

    We store:
    - vectors in FAISS (fast ANN similarity search)
    - chunk text + IDs in SQLite (durable, queryable)

    Similarity:
    - SentenceTransformers vectors are normalized
    - We use IndexFlatIP (inner product) which equals cosine similarity for normalized vectors.
    """

    def __init__(self, store_dir: str | Path) -> None:
        self._dir = ensure_dir(store_dir)
        self._index_path = self._dir / "index.faiss"
        self._db_path = self._dir / "metadata.db"
        self._lock = threading.Lock()

        self._index: faiss.Index | None = None
        self._dim: int | None = None

        self._init_db()
        self._load_index_if_exists()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                  faiss_id INTEGER PRIMARY KEY,
                  chunk_id TEXT NOT NULL UNIQUE,
                  document_id TEXT NOT NULL,
                  chunk_text TEXT NOT NULL,
                  start_token INTEGER NOT NULL,
                  end_token INTEGER NOT NULL
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id);")
            conn.commit()

            # Detect legacy schema (row_id-based) and rebuild to avoid FAISS<->SQLite mismatches.
            cols = [r[1] for r in conn.execute("PRAGMA table_info(chunks);").fetchall()]
            if "row_id" in cols and "faiss_id" not in cols:
                self._rebuild_store(conn)
        finally:
            conn.close()

    def _load_index_if_exists(self) -> None:
        if self._index_path.exists():
            self._index = faiss.read_index(str(self._index_path))
            self._dim = self._index.d
            # Ensure we can store explicit ids
            if not isinstance(self._index, faiss.IndexIDMap):
                # Wrap legacy index (no IDs) into IDMap; ids must then be rebuilt.
                self._index = faiss.IndexIDMap2(self._index)

    def _ensure_index(self, dim: int) -> None:
        if self._index is None:
            # Flat index is exact search; for larger scale you can swap to IVF/HNSW.
            self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
            self._dim = dim
        elif self._dim != dim:
            raise ValueError(f"Embedding dimension mismatch: existing={self._dim}, new={dim}")

    def add_embeddings(
        self,
        *,
        document_id: str,
        chunk_ids: list[str],
        chunk_texts: list[str],
        start_tokens: list[int],
        end_tokens: list[int],
        embeddings: np.ndarray,
    ) -> int:
        if len(chunk_ids) != len(chunk_texts) or len(chunk_ids) != embeddings.shape[0]:
            raise ValueError("Mismatched chunk metadata lengths")

        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be 2D [n, dim]")

        with self._lock:
            self._ensure_index(dim=int(embeddings.shape[1]))

            # Allocate stable FAISS ids and persist metadata (durable mapping).
            conn = self._connect()
            try:
                max_id_row = conn.execute("SELECT COALESCE(MAX(faiss_id), -1) FROM chunks;").fetchone()
                start_id = int(max_id_row[0]) + 1
                ids = np.arange(start_id, start_id + len(chunk_ids), dtype=np.int64)

                conn.executemany(
                    """
                    INSERT INTO chunks(faiss_id, chunk_id, document_id, chunk_text, start_token, end_token)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    list(
                        zip(
                            ids.tolist(),
                            chunk_ids,
                            [document_id] * len(chunk_ids),
                            chunk_texts,
                            start_tokens,
                            end_tokens,
                        )
                    ),
                )
                conn.commit()
            finally:
                conn.close()

            # Add vectors with explicit ids so retrieval is correct and stable.
            assert isinstance(self._index, faiss.IndexIDMap)
            self._index.add_with_ids(embeddings, ids)
            faiss.write_index(self._index, str(self._index_path))

        return len(chunk_ids)

    def _fetch_chunk_by_faiss_id(self, conn: sqlite3.Connection, faiss_id: int) -> tuple[str, str, str]:
        row = conn.execute(
            """
            SELECT document_id, chunk_id, chunk_text
            FROM chunks
            WHERE faiss_id = ?
            """,
            (faiss_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Chunk id {faiss_id} not found")
        return str(row[0]), str(row[1]), str(row[2])

    def search(
        self,
        query_embedding: np.ndarray,
        *,
        top_k: int = 4,
        document_ids: list[str] | None = None,
        oversample: int = 30,
    ) -> list[RetrievedChunk]:
        """
        Search by cosine similarity (inner product on normalized vectors).

        Multi-document support:
        - If document_ids is provided, we oversample and filter by doc id.
        """

        if self._index is None or self._index.ntotal == 0:
            return []

        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)
        q = query_embedding.reshape(1, -1)

        k = min(max(top_k, 1), 10)
        k_search = min(self._index.ntotal, max(k, oversample if document_ids else k))

        with self._lock:
            scores, ids = self._index.search(q, k_search)

        conn = self._connect()
        try:
            results: list[RetrievedChunk] = []
            wanted = set(document_ids or [])
            for score, idx in zip(scores[0].tolist(), ids[0].tolist()):
                if idx < 0:
                    continue
                doc_id, chunk_id, chunk_text = self._fetch_chunk_by_faiss_id(conn, int(idx))
                if document_ids is not None and doc_id not in wanted:
                    continue
                results.append(
                    RetrievedChunk(document_id=doc_id, chunk_id=chunk_id, score=float(score), text=chunk_text)
                )
                if len(results) >= k:
                    break
            return results
        finally:
            conn.close()

    def _rebuild_store(self, conn: sqlite3.Connection) -> None:
        """
        Legacy schema safety:
        - If an old (row_id-based) schema exists, the FAISS<->SQLite mapping will be incorrect.
        - Rebuild into faiss_id-based schema so search results return the correct chunk texts.
        """

        # Backup old DB and index (best-effort).
        try:
            if self._db_path.exists():
                self._db_path.rename(self._dir / "metadata.db.bak")
            if self._index_path.exists():
                self._index_path.rename(self._dir / "index.faiss.bak")
        except Exception:
            pass

        # Drop and recreate with the correct schema.
        conn.execute("DROP TABLE IF EXISTS chunks;")
        conn.execute(
            """
            CREATE TABLE chunks (
              faiss_id INTEGER PRIMARY KEY,
              chunk_id TEXT NOT NULL UNIQUE,
              document_id TEXT NOT NULL,
              chunk_text TEXT NOT NULL,
              start_token INTEGER NOT NULL,
              end_token INTEGER NOT NULL
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id);")
        conn.commit()

        # Reset in-memory index
        self._index = None
        self._dim = None

