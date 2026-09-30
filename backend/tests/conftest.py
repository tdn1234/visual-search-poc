"""Shared fixtures.

The environment overrides below MUST run before anything imports `app.*`:
`app.config` reads them at import time, and `app.rate_limit` builds its
Redis-backed `Limiter` from `REDIS_URL` at import time -- so without
this, the API tests would try to reach a real Redis.
"""

from __future__ import annotations

import io
import os

# In-memory rate-limit storage, and limits that are effectively off --
# except /search, kept low so the 429 test only needs a handful of calls.
os.environ["REDIS_URL"] = "memory://"
os.environ["SEARCH_RATE_LIMIT"] = "5/minute"
os.environ["PRODUCTS_RATE_LIMIT"] = "1000/minute"
os.environ["CREATE_PRODUCT_RATE_LIMIT"] = "1000/minute"
os.environ["BULK_IMPORT_RATE_LIMIT"] = "1000/minute"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from app.config import API_KEY  # noqa: E402
from app.main import app  # noqa: E402
from app.rate_limit import limiter  # noqa: E402
from tests.fakes import (  # noqa: E402
    FakeAttributeClassifier,
    FakeEmbeddingService,
    FakeIndexingService,
    FakeProductQueryService,
)


def _image_bytes(fmt: str) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (200, 30, 30)).save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.fixture
def png_bytes() -> bytes:
    return _image_bytes("PNG")


@pytest.fixture
def jpeg_bytes() -> bytes:
    return _image_bytes("JPEG")


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Rate-limit counters live in process memory; don't let tests leak into each other."""
    limiter.reset()
    yield
    limiter.reset()


@pytest.fixture
def auth_headers() -> dict[str, str]:
    return {"X-API-Key": API_KEY}


@pytest.fixture
def services():
    """Fakes attached to the real `app.state`, exactly where `main.lifespan` would put the real ones."""
    fakes = type(
        "Services",
        (),
        {
            "embedding": FakeEmbeddingService(),
            "product_query": FakeProductQueryService(),
            "attribute_classifier": FakeAttributeClassifier(),
            "indexing": FakeIndexingService(),
        },
    )()
    app.state.embedding_service = fakes.embedding
    app.state.product_query_service = fakes.product_query
    app.state.attribute_classifier_service = fakes.attribute_classifier
    app.state.indexing_service = fakes.indexing
    return fakes


@pytest.fixture
def client(services) -> TestClient:
    # Not used as a context manager on purpose: that would run
    # `lifespan`, which loads CLIP and connects to Postgres.
    return TestClient(app)
