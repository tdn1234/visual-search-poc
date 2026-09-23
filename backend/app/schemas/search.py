"""Pydantic schemas shared across services and API routes.

Keeping these in one module means the "shape" of a product and a
search result is defined exactly once.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ProductRecord(BaseModel):
    """A single indexed product: metadata + its precomputed embedding.

    This is the shape of each entry stored in `backend/data/embeddings.json`.
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
