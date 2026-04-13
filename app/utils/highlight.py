from __future__ import annotations

import re


def highlight_terms(text: str, query: str, *, max_terms: int = 8) -> str:
    """
    Simple, dependency-free source highlighting for UX:
    - Extracts keyword-ish terms from the query
    - Wraps occurrences in <mark>...</mark>
    """

    terms = _extract_terms(query)[:max_terms]
    if not terms:
        return text

    pattern = re.compile(r"(" + "|".join(re.escape(t) for t in terms) + r")", re.IGNORECASE)
    return pattern.sub(r"<mark>\1</mark>", text)


def _extract_terms(q: str) -> list[str]:
    q = q.strip().lower()
    tokens = re.findall(r"[a-z0-9]{3,}", q)
    stop = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "how",
        "into",
        "are",
        "was",
        "were",
        "will",
        "would",
        "could",
        "should",
        "can",
        "may",
        "might",
        "your",
        "you",
        "about",
    }
    uniq: list[str] = []
    for t in tokens:
        if t in stop:
            continue
        if t not in uniq:
            uniq.append(t)
    return uniq

