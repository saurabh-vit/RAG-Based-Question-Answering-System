from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Request, UploadFile

from app.models.upload import UploadResponse
from app.services.container import Container
from app.utils.limiter import limiter
from app.utils.settings import get_settings
from app.utils.files import ensure_dir, extension_ok, new_document_id, safe_filename


log = logging.getLogger("rag.routes.upload")
router = APIRouter(tags=["upload"])
settings = get_settings()


def _get_container(request: Request) -> Container:
    return request.app.state.container


@router.post("/upload", response_model=UploadResponse)
@limiter.limit(settings.rate_limit_upload)
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    if not file.filename or not extension_ok(file.filename):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")

    container = _get_container(request)
    settings = container.settings

    doc_id = new_document_id()
    data_dir = ensure_dir(settings.data_dir)
    doc_dir = ensure_dir(data_dir / doc_id)

    filename = safe_filename(file.filename)
    path = doc_dir / filename

    # Save file locally (streaming to avoid memory spikes)
    try:
        with path.open("wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    finally:
        await file.close()

    # Async ingestion (in-process background task).
    # For higher scale, swap to Celery/RQ; the ingestion service is already pure.
    background_tasks.add_task(container.ingestion_service.ingest_document, document_id=doc_id, file_path=str(path))

    log.info("Upload accepted", extra={"document_id": doc_id, "file": str(Path(path).name)})
    return UploadResponse(document_id=doc_id)

