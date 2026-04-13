from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


def extract_text_from_file(path: str | Path) -> str:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".txt":
        # UTF-8 is the default for production; you can extend this with chardet if needed.
        return p.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        reader = PdfReader(str(p))
        parts: list[str] = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        return "\n".join(parts)
    raise ValueError(f"Unsupported file type: {suffix}")

