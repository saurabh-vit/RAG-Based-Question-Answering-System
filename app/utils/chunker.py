from __future__ import annotations

from dataclasses import dataclass

import tiktoken


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    start_token: int
    end_token: int


def chunk_text_tokenwise(
    text: str,
    *,
    document_id: str,
    chunk_size_tokens: int = 400,
    overlap_tokens: int = 80,
    encoding_name: str = "cl100k_base",
) -> list[Chunk]:
    """
    Token-based chunking (not character-based) gives more predictable LLM context usage.

    Why 300–500 tokens with 50–100 overlap?
    - **300–500 tokens**: large enough to capture full paragraphs + local definitions (better retrieval coherence),
      small enough to keep high recall and allow multiple chunks in the prompt without blowing context length.
    - **50–100 overlap**: reduces "boundary loss" (facts split across chunk edges) at a modest storage cost.

    Trade-offs:
    - Smaller chunks: higher recall, more vectors, more FAISS entries, more prompt assembly overhead; can hurt coherence.
    - Larger chunks: fewer vectors and cheaper indexing, but lower recall (needle facts get diluted), higher chance of
      retrieving irrelevant text around the answer.
    """

    enc = tiktoken.get_encoding(encoding_name)
    tokens = enc.encode(text)

    if chunk_size_tokens <= 0:
        raise ValueError("chunk_size_tokens must be > 0")
    if overlap_tokens < 0 or overlap_tokens >= chunk_size_tokens:
        raise ValueError("overlap_tokens must be >=0 and < chunk_size_tokens")

    chunks: list[Chunk] = []
    start = 0
    n = len(tokens)
    idx = 0
    while start < n:
        end = min(start + chunk_size_tokens, n)
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens).strip()
        if chunk_text:
            chunk_id = f"{document_id}_{idx}"
            chunks.append(Chunk(chunk_id=chunk_id, text=chunk_text, start_token=start, end_token=end))
            idx += 1
        if end >= n:
            break
        start = end - overlap_tokens

    return chunks

