from __future__ import annotations

import os
import uuid
from pathlib import Path


ALLOWED_EXTENSIONS = {".pdf", ".txt"}


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def new_document_id() -> str:
    # URL-safe-ish id; easy to log and copy around
    return uuid.uuid4().hex


def safe_filename(original: str) -> str:
    name = os.path.basename(original)
    name = name.replace("..", ".")
    return name


def extension_ok(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

