from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # rag-system/
_ENV_FILE = _PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    """
    Central configuration for the service.

    We keep this in one place so that routes/services remain pure and testable.
    """

    model_config = SettingsConfigDict(env_file=str(_ENV_FILE), env_file_encoding="utf-8", extra="ignore")

    app_name: str = Field(default="rag-system", alias="APP_NAME")
    env: str = Field(default="dev", alias="ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Paths are resolved relative to the project root (rag-system/),
    # so the app behaves the same whether you launch uvicorn from repo root or from rag-system/.
    data_dir: str = Field(default="data", alias="DATA_DIR")
    vector_store_dir: str = Field(default="vector_store", alias="VECTOR_STORE_DIR")

    embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", alias="EMBEDDING_MODEL_NAME"
    )

    top_k: int = Field(default=4, alias="TOP_K")

    # LLM
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4.1-mini", alias="OPENAI_MODEL")

    # Rate limits (slowapi syntax)
    rate_limit_upload: str = Field(default="5/minute", alias="RATE_LIMIT_UPLOAD")
    rate_limit_ask: str = Field(default="10/minute", alias="RATE_LIMIT_ASK")

    # Chunking defaults
    chunk_size_tokens: int = Field(default=400, alias="CHUNK_SIZE_TOKENS")
    chunk_overlap_tokens: int = Field(default=80, alias="CHUNK_OVERLAP_TOKENS")

    # Cache
    cache_ttl_seconds: int = Field(default=120, alias="CACHE_TTL_SECONDS")


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.data_dir = str(_resolve_project_path(_settings.data_dir))
        _settings.vector_store_dir = str(_resolve_project_path(_settings.vector_store_dir))
    return _settings


def _resolve_project_path(p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    # settings.py -> utils/ -> app/ -> rag-system/
    project_root = Path(__file__).resolve().parents[2]
    return project_root / path

