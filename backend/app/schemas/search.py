"""Pydantic schemas shared across services and API routes.

Keeping these in one module means the "shape" of a product and a
search result is defined exactly once.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.config import (
    CATEGORY_MAX_LENGTH,
    COLOR_MAX_LENGTH,
    MAX_PRICE,
    NAME_MAX_LENGTH,
    SKU_PATTERN,
    TEXT_PATTERN,
)


class ProductRecord(BaseModel):
    """A single indexed product: metadata + its precomputed embedding.

    This is the shape of each row in the Postgres `products` table.
    """

    sku: str = Field(..., description="Unique product identifier, e.g. 'shoe-red'.")
    name: str = Field(..., description="Human-readable product name.")
    price: float = Field(..., description="Product price.")
    category: str = Field(..., description="Product category, e.g. 'Shoes'.")
    color: str | None = Field(None, description="Dominant product color, e.g. 'brown'. Optional.")
    image_path: str = Field(..., description="Path to the source image, relative to the project root.")
    embedding: list[float] = Field(..., description="L2-normalized CLIP image embedding.")


class SearchResult(BaseModel):
    """A single ranked match returned by the /search endpoint."""

    sku: str
    name: str
    price: float
    category: str
    score: float = Field(..., description="Cosine similarity score in [-1, 1]; higher is more similar.")


class SearchResponse(BaseModel):
    """Top-level response envelope for POST /search."""

    results: list[SearchResult]


class ProductSummary(BaseModel):
    """A catalog product without its embedding, for browsing/filtering."""

    sku: str
    name: str
    price: float
    category: str
    color: str | None = None


class ProductListResponse(BaseModel):
    """Top-level response envelope for GET /products."""

    results: list[ProductSummary]


class BulkProductItem(BaseModel):
    """One product's metadata inside a `POST /products/import` batch.

    Same fields/constraints as the single-product `POST /products`
    form, just as JSON instead of form fields -- see
    `api/products.py`'s `bulk_import_products` for how the matching
    `files` list lines up with these by array index.
    """

    sku: str = Field(..., pattern=SKU_PATTERN, description="Unique product identifier, e.g. 'shoe-purple'.")
    name: str = Field(..., max_length=NAME_MAX_LENGTH, pattern=TEXT_PATTERN, description="Human-readable product name.")
    price: float = Field(..., gt=0, le=MAX_PRICE, allow_inf_nan=False, description="Product price.")
    category: str = Field(..., max_length=CATEGORY_MAX_LENGTH, pattern=TEXT_PATTERN, description="Product category, e.g. 'Shoes'.")
    color: str | None = Field(
        None, max_length=COLOR_MAX_LENGTH, pattern=TEXT_PATTERN, description="Dominant product color, e.g. 'purple'. Optional."
    )


class BulkImportResponse(BaseModel):
    """Top-level response envelope for POST /products/import.

    Fire-and-forget by design (see docs/architecture.md's "Bulk
    product import" section) -- this only confirms the batch was
    queued, not that any individual product was actually imported.
    Per-item outcomes are in the worker's logs, correlated by
    `batch_id` (e.g. `docker-compose logs worker | grep <batch_id>`).
    """

    batch_id: str
    queued: int
