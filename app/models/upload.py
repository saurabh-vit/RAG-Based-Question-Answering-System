from __future__ import annotations

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    document_id: str = Field(..., description="ID used for querying and tracking ingestion.")
    status: str = Field(default="accepted", description="Upload status; ingestion runs asynchronously.")

