from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from starlette.responses import JSONResponse

from app.routes.query import router as query_router
from app.routes.upload import router as upload_router
from app.services.container import Container
from app.utils.limiter import limiter
from app.utils.logging import setup_logging
from app.utils.metrics import Timer
from app.utils.settings import get_settings
from dotenv import load_dotenv

def create_app() -> FastAPI:
    # Load environment variables from rag-system/.env (if present).
    # Settings also loads from .env, but load_dotenv supports local dev parity.
    load_dotenv()

    settings = get_settings()
    setup_logging(settings.log_level)
    log = logging.getLogger("rag.app")

    app = FastAPI(title=settings.app_name)

    # CORS: allow local UI + dev clients; tighten in production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Rate limiting
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

    # Dependency container
    app.state.container = Container(settings)

    # Metrics middleware (latency)
    @app.middleware("http")
    async def latency_middleware(request: Request, call_next: Callable[[Request], Awaitable[Any]]):
        timer = Timer.start_new()
        response: Any = None
        try:
            response = await call_next(request)
            return response
        finally:
            latency_ms = timer.elapsed_ms()
            # Add header for quick debugging / dashboards
            if response is not None:
                response.headers["X-Latency-Ms"] = f"{latency_ms:.2f}"
            log.info(
                "request_complete",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "latency_ms": round(latency_ms, 2),
                },
            )

    # Health
    @app.get("/health")
    async def health():
        return {"status": "ok"}

    # Root (required for your audit)
    @app.get("/")
    async def root():
        return {"message": "RAG API is running", "docs": "/docs"}

    # Routers with per-route rate limits (slowapi decorator reads app.state.limiter)
    app.include_router(upload_router)
    app.include_router(query_router)

    # Custom 500 response (avoid leaking internals)
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(_: Request, exc: Exception):
        log.exception("unhandled_exception", extra={"error": str(exc)})
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    log.info("app_started", extra={"env": settings.env})
    return app


app = create_app()

