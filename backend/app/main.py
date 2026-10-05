"""FastAPI application entry point.

Responsibility: create the FastAPI app, load the (heavy) CLIP model
exactly once at startup, open the Postgres connection pool, wire up
auth/rate-limiting, store everything on `app.state`, and register API
routes. The product catalog itself is *not* loaded here -- every
request queries Postgres directly (see `ProductQueryService`), so this
module only owns things that are genuinely expensive to set up
per-request (the ML models, the DB pool).

Run with:
    uvicorn app.main:app --reload
(from the `backend/` directory, with the virtual environment active).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.api.products import router as products_router
from app.api.recommendations import router as recommendations_router
from app.api.search import router as search_router
from app.config import API_KEY, CATEGORY_CLASSIFIER_FILE, COLOR_ADAPTER_FILE, MAX_REQUEST_BODY_BYTES
from app.db import create_pool
from app.logging_config import configure_logging
from app.middleware import BodySizeLimitMiddleware, RequestContextMiddleware
from app.models.category_classifier import load_category_classifier
from app.models.clip_model import ClipModel
from app.models.color_adapter import load_color_adapter
from app.rate_limit import limiter
from app.services.attribute_classifier_service import AttributeClassifierService
from app.services.embedding_service import EmbeddingService
from app.services.indexing_service import IndexingService
from app.services.product_query_service import ProductQueryService
from app.services.recommendation_service import RecommendationService

configure_logging()
logger = logging.getLogger(__name__)

_DEFAULT_API_KEY = "api-key"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ML models and open the DB pool once, before serving requests.

    Loading CLIP takes a few seconds and must not happen per-request,
    so it lives here and the resulting objects are attached to
    `app.state` for the API routes to reuse.
    """
    if API_KEY == _DEFAULT_API_KEY:
        logger.warning(
            "Using the default dev API key. Set the API_KEY env var before exposing "
            "this service beyond your own machine."
        )

    logger.info("Starting up: connecting to Postgres...")
    db_pool = create_pool()

    logger.info("Loading CLIP model...")
    clip_model = ClipModel()
    color_adapter = load_color_adapter(COLOR_ADAPTER_FILE)
    category_classifier = load_category_classifier(CATEGORY_CLASSIFIER_FILE)

    embedding_service = EmbeddingService(clip_model=clip_model, color_adapter=color_adapter)
    product_query_service = ProductQueryService(db_pool=db_pool)
    indexing_service = IndexingService(embedding_service=embedding_service, db_pool=db_pool)
    recommendation_service = RecommendationService(db_pool=db_pool)
    attribute_classifier_service = AttributeClassifierService(
        clip_model=clip_model, category_classifier=category_classifier
    )

    logger.info("%d products currently indexed in Postgres", product_query_service.count())

    app.state.db_pool = db_pool
    app.state.embedding_service = embedding_service
    app.state.product_query_service = product_query_service
    app.state.indexing_service = indexing_service
    app.state.attribute_classifier_service = attribute_classifier_service
    app.state.recommendation_service = recommendation_service

    yield

    logger.info("Shutting down.")
    db_pool.close()


app = FastAPI(
    title="Visual Product Search POC",
    description="Upload a product image and get the top visually similar catalog items.",
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(BodySizeLimitMiddleware, max_bytes=MAX_REQUEST_BODY_BYTES)
# Added last so it becomes the *outermost* middleware (Starlette wraps in
# reverse registration order): the request ID must be assigned before
# rate limiting runs, and the access-log line must see the real final
# status code (200/401/429/...), not just whatever SlowAPIMiddleware saw.
app.add_middleware(RequestContextMiddleware)

app.include_router(search_router)
app.include_router(products_router)
app.include_router(recommendations_router)


@app.get("/health", tags=["health"])
async def health_check() -> dict[str, str]:
    """Simple liveness/readiness check for local development."""
    return {"status": "ok"}
