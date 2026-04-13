from __future__ import annotations

import logging
import os
from typing import Iterable

from dotenv import load_dotenv
import google.generativeai as genai

log = logging.getLogger("rag.llm")

PREFERRED_MODEL_NAME = "gemini-1.0-pro"

def _configure_gemini() -> None:
    # Load .env for local dev + ensure env vars are available inside this module too.
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    log.info("gemini_key_loaded", extra={"loaded": bool(api_key)})
    log.info("gemini_model_preferred", extra={"model": PREFERRED_MODEL_NAME})
    if api_key:
        # SDK is dynamically typed; keep runtime behavior correct.
        genai.configure(api_key=api_key)  # type: ignore[attr-defined]


def _pick_supported_model_name() -> str:
    """
    Gemini model availability varies by account/region/API version.
    We *prefer* PREFERRED_MODEL_NAME (per your requirement), but we must fall back
    to a model that actually supports `generateContent`, otherwise the system cannot run.
    """

    try:
        models = list(genai.list_models())  # type: ignore[attr-defined]
    except Exception as e:
        log.exception("gemini_list_models_failed", extra={"error": str(e)})
        return PREFERRED_MODEL_NAME

    supported: list[str] = []
    for m in models:  # type: ignore[unknown-variable-type]
        name = getattr(m, "name", None)  # type: ignore[arg-type]
        methods = getattr(m, "supported_generation_methods", None) or []  # type: ignore[arg-type]
        if name and any(str(x).lower() == "generatecontent" for x in methods):  # type: ignore[unknown-variable-type]
            supported.append(str(name))

    if not supported:
        return PREFERRED_MODEL_NAME

    # Try preferred first (matches either "models/<name>" or "<name>")
    for cand in supported:
        if cand.endswith("/" + PREFERRED_MODEL_NAME) or cand == PREFERRED_MODEL_NAME:
            return cand

    # Prefer flash, then pro, else first supported.
    for key in ("flash", "pro"):
        for cand in supported:
            if key in cand.lower():
                return cand

    return supported[0]


def _get_model():
    _configure_gemini()
    model_name = _pick_supported_model_name()
    log.info("gemini_model_selected", extra={"model": model_name})
    return genai.GenerativeModel(model_name)  # type: ignore[attr-defined]


def build_context(chunks: Iterable[str], *, max_chars: int = 12000) -> str:
    parts: list[str] = []
    total = 0
    for i, c in enumerate(chunks, start=1):
        block = f"[Source {i}]\n{c.strip()}\n"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n".join(parts).strip()


def generate_answer(context: str, question: str) -> str:
    # Temporary debug logs (do not print the key itself).
    log.info("gemini_context_length", extra={"context_length": len(context or "")})

    if not (context or "").strip():
        return "Not in document."

    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        return (
            "Missing Google AI Studio key. "
            "Please provide your Google AI Studio API key to proceed. "
            "Add it in `.env` as: GOOGLE_API_KEY=your_key_here"
        )

    try:
        prompt = f"""
You are an AI assistant.
Answer the question ONLY using the provided context.

Context:
{context}

Question:
{question}

Give a short, clear, and factual answer.
"""

        model = _get_model()
        response = model.generate_content(prompt)  # type: ignore[no-untyped-call]
        return (response.text or "").strip()

    except Exception as e:
        log.exception("gemini_generate_failed", extra={"error": str(e)})
        return f"Error generating answer: {str(e)}"


class LLMService:
    """
    Compatibility wrapper so the rest of the app can keep calling `container.llm_service.answer(...)`.
    """

    def answer(self, *, question: str, context: str) -> str:
        return generate_answer(context=context, question=question)

