# RAG-Based Question Answering System

## 1. Project Title

**RAG-Based Question Answering System**

---

## 2. Overview

### What is RAG?
**Retrieval-Augmented Generation (RAG)** is a system design pattern that combines:
- **Retrieval**: find the most relevant parts of your documents (via embeddings + vector search)
- **Generation**: ask a language model (LLM) to answer **using only** the retrieved context

### What problem does this project solve?
LLMs are powerful but can hallucinate. This project makes answers **grounded in your uploaded documents** by:
- turning documents into searchable embeddings
- retrieving the best-matching chunks at query time
- generating a short factual answer from those chunks using **Google Gemini**

### Real-world use cases
- Internal knowledge base (HR policies, onboarding docs, engineering runbooks)
- Research assistant for PDFs (reports, papers, proposals)
- Customer support FAQ over product manuals and terms

---

## 3. Features

- **Document upload**: `POST /upload` supports **PDF** and **TXT**
- **Asynchronous ingestion**: uses FastAPI **BackgroundTasks** (API stays responsive)
- **Chunking + embeddings**: token-aware chunking and local embeddings via **SentenceTransformers**
- **FAISS vector search**: fast similarity search for top relevant chunks
- **Gemini-based answer generation**: uses retrieved context to produce a short answer (not raw chunks)
- **Source attribution**: returns sources with **similarity scores** and chunk IDs
- **Latency tracking**: returns `latency_ms` in API response and `X-Latency-Ms` header

---

## 4. Tech Stack (with justification)

- **FastAPI**
  - Clean async-friendly API framework with auto OpenAPI docs (`/docs`)
  - Works well with background ingestion pipelines
- **SentenceTransformers**
  - High-quality **local** embeddings (no dependency on remote embedding APIs)
  - Good balance of performance and accuracy for document retrieval
- **FAISS**
  - Industry-standard vector similarity search library
  - Local-first and fast for small/medium datasets
- **Google Gemini API**
  - Generates concise answers from retrieved context
  - Separates retrieval from generation for grounded responses
- **Pydantic**
  - Strong request/response validation and clear API contracts
- **python-dotenv**
  - Loads `.env` for local development (API keys and settings without hardcoding secrets)

---

## 5. System Architecture

```mermaid
flowchart LR
  U[User/Client] -->|POST /upload| API[FastAPI]
  API -->|Save file| FS[(data/)]
  API -->|BackgroundTasks| ING[Ingestion Pipeline]
  ING --> EX[Text Extraction\nPDF/TXT]
  EX --> CH[Chunking\n400 tokens, overlap 80]
  CH --> EMB[Embeddings\nSentenceTransformers]
  EMB --> VS[(FAISS Vector Store)]

  U -->|POST /ask| API
  API --> QEMB[Query Embedding]
  QEMB --> VS
  VS --> RET[Top-K Chunks\n+ similarity scores]
  RET --> LLM[Gemini Generate\n(Grounded Answer)]
  LLM --> API
  API --> U
```

---

## 6. Project Structure

```text
rag-system/
├─ app/
│  ├─ main.py                  # FastAPI app bootstrap + middleware
│  ├─ routes/                  # HTTP layer (request/response)
│  │  ├─ upload.py             # POST /upload
│  │  └─ query.py              # POST /ask
│  ├─ services/                # Core business logic
│  │  ├─ ingestion_service.py  # Extract → chunk → embed → store
│  │  ├─ embedding_service.py  # SentenceTransformer wrapper
│  │  ├─ vector_store.py       # FAISS index + metadata store
│  │  ├─ llm_service.py        # Gemini answer generation
│  │  └─ container.py          # Dependency wiring
│  ├─ models/                  # Pydantic schemas
│  └─ utils/                   # Shared utilities (chunker, settings, logging)
├─ data/                       # Uploaded documents (per document_id)
├─ vector_store/               # Local FAISS index + metadata
├─ requirements.txt
├─ .env.example
└─ README.md
```

**Design note**: routes are thin. Most logic lives in services/utils to keep the code maintainable and testable.

---

## 7. Setup Instructions

### 1) Clone the repository

```bash
git clone <your-repo-url>
cd <your-repo>
```

### 2) Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3) Install dependencies

```bash
pip install -r rag-system/requirements.txt
```

### 4) Create `.env`

Copy `.env.example` → `.env`:

```bash
copy rag-system\.env.example rag-system\.env
```

### 5) Add your Google AI Studio key

Edit `rag-system/.env`:

```env
GOOGLE_API_KEY=your_key_here
```

### 6) Run the server

From the repo root:

```bash
python -m uvicorn app.main:app --app-dir rag-system --host 127.0.0.1 --port 8000
```

OpenAPI docs:
- `http://127.0.0.1:8000/docs`

---

## 8. API Endpoints

### POST `/upload`

- **Input**: multipart form-data file (`.pdf` or `.txt`)
- **Output**: `document_id` (used to query and filter retrieval)

Example:

```bash
curl -X POST "http://127.0.0.1:8000/upload" ^
  -F "file=@C:\path\to\document.txt"
```

Response:

```json
{
  "document_id": "e7a3...c21",
  "status": "accepted"
}
```

### POST `/ask`

- **Input**:
  - `question` (string)
  - `document_ids` (list of document IDs; optional but recommended)
  - `top_k` (int; optional, min 3 enforced)
- **Output**:
  - `answer` (Gemini-generated)
  - `sources` (retrieved chunks with similarity score)
  - `latency_ms`

Request:

```json
{
  "question": "How many recruitment agencies are listed?",
  "document_ids": ["REAL_DOCUMENT_ID"],
  "top_k": 3
}
```

Response shape:

```json
{
  "answer": "50 recruitment agencies are listed in the document.",
  "sources": [
    {
      "document_id": "REAL_DOCUMENT_ID",
      "chunk_id": "REAL_DOCUMENT_ID_0",
      "score": 0.77,
      "text": "…",
      "highlighted_text": "…"
    }
  ],
  "cached": false,
  "latency_ms": 42.1
}
```

---

## 9. Example Usage (realistic end-to-end)

1) Upload a TXT:

```bash
curl -X POST "http://127.0.0.1:8000/upload" ^
  -F "file=@C:\docs\agencies.txt"
```

2) Ask a question:

```bash
curl -X POST "http://127.0.0.1:8000/ask" ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"How many recruitment agencies are listed?\",\"document_ids\":[\"<document_id>\"],\"top_k\":3}"
```

---

## 10. Chunking Strategy (MANDATORY)

This project uses **token-based chunking**:
- **Chunk size**: **400 tokens**
- **Overlap**: **80 tokens**

### Why this improves retrieval quality
- **Token-sized chunks** map better to LLM context usage than character splitting.
- **400 tokens** usually captures a full paragraph + local definitions.
- **Overlap (80 tokens)** reduces boundary loss (important facts split across chunk edges).

Trade-off:
- More overlap ⇒ slightly more storage + embeddings, but improved recall.

---

## 11. Retrieval Failure Case (MANDATORY)

### Example failure
Question: **“How many agencies are listed?”**

If the document contains multiple sections with “agency” (e.g., marketing agencies, recruitment agencies, travel agencies),
retrieval may pull a chunk from the wrong section.

### Why it failed
- Query is **too broad** (weak intent signal)
- Similar terms exist across unrelated sections

### How to improve
- Add **hybrid retrieval** (BM25 + embeddings)
- Add **re-ranking** (cross-encoder)
- Add **metadata** (section titles, headings) and include them in chunks

---

## 12. Metrics Tracked (MANDATORY)

- **Latency**
  - Returned as `latency_ms` and `X-Latency-Ms`
  - Helps detect slow embedding/model calls and measure performance
- **Similarity score**
  - Returned per source chunk (`score`)
  - Helps debug retrieval quality and tune chunking/top_k

Why these matter:
- In production RAG, performance and retrieval quality determine user trust and cost.

---

## 13. Limitations

- **LLM dependency**: answer generation requires a valid Google AI Studio key and available Gemini models.
- **Retrieval mismatch**: broad questions can retrieve the wrong section.
- **Latency variance**: first-run downloads (embedding model) and network calls (Gemini) can spike response time.

---

## 14. Future Improvements

- Better ranking:
  - hybrid retrieval (BM25 + embeddings)
  - re-ranking model
- Caching:
  - cache embeddings for repeated queries
  - cache Gemini answers for repeated questions
- UI:
  - improve Streamlit UI and add document management
- Multi-document reasoning:
  - stronger cross-document retrieval and summarization

---

## 15. Conclusion

This project is a **production-style RAG backend** that supports:
- document upload and asynchronous ingestion
- token-aware chunking and local embeddings
- FAISS similarity search with source attribution
- **Gemini-generated grounded answers**
- latency and similarity observability for debugging and tuning

# RAG-Based-Question-Answering-System
# RAG-Based-Question-Answering-System
