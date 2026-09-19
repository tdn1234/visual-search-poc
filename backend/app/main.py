"""FastAPI application entry point.

Responsibility: create the FastAPI app, load the (heavy) CLIP model
and the product index exactly once at startup, store them on
`app.state`, and register API routes.

Run with:
    uvicorn app.main:app --reload
(from the `backend/` directory, with the virtual environment active).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.search import router as search_router
from app.config import EMBEDDINGS_FILE
from app.models.clip_model import ClipModel
from app.services.embedding_service import EmbeddingService
from app.services.indexing_service import IndexingService
from app.services.similarity_service import SimilarityService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ML model and product catalog once, before serving requests.

    Loading CLIP takes a few seconds and must not happen per-request,
    so it lives here and the resulting objects are attached to
    `app.state` for the API routes to reuse.
    """
    logger.info("Starting up: loading CLIP model...")
    clip_model = ClipModel()

    embedding_service = EmbeddingService(clip_model=clip_model)
    indexing_service = IndexingService(embedding_service=embedding_service)
    similarity_service = SimilarityService()

    catalog = indexing_service.load_index(EMBEDDINGS_FILE)
    logger.info("Loaded %d products from %s", len(catalog), EMBEDDINGS_FILE)

    app.state.embedding_service = embedding_service
    app.state.indexing_service = indexing_service
    app.state.similarity_service = similarity_service
    app.state.catalog = catalog

    yield

    logger.info("Shutting down.")


app = FastAPI(
    title="Visual Product Search POC",
    description="Upload a product image and get the top visually similar catalog items.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(search_router)


@app.get("/health", tags=["health"])
async def health_check() -> dict[str, str]:
    """Simple liveness/readiness check for local development."""
    return {"status": "ok"}
