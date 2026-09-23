"""HTTP API layer for filtering the catalog by category and/or color.

Responsibility: exact-metadata filtering over the in-memory catalog.
This is deliberately separate from `api/search.py` -- there's no
image involved and no embedding/similarity math, just a plain filter
over fields already sitting in `ProductRecord`.
"""

from __future__ import annotations

from fastapi import APIRouter, Query, Request

from app.schemas.search import ProductListResponse, ProductSummary

router = APIRouter(tags=["products"])


@router.get("/products", response_model=ProductListResponse)
async def list_products(
    request: Request,
    category: str | None = Query(None, description="Exact category match, e.g. 'Shoes'."),
    color: str | None = Query(None, description="Exact color match, e.g. 'red'."),
) -> ProductListResponse:
    """List catalog products, optionally filtered by category and/or color.

    Matching is case-insensitive but exact (not fuzzy/partial) against
    each product's `category`/`color` metadata. A product with no
    `color` set never matches a `color` filter. Passing neither filter
    returns the whole catalog.

    Args:
        request: The FastAPI request, used to reach the in-memory
            catalog stored on `app.state` (set up once at startup).
        category: If given, only return products in this category.
        color: If given, only return products with this color.

    Returns:
        A `ProductListResponse` listing every matching product (no
        embeddings, no ranking/score -- this is a plain filter).
    """
    catalog = request.app.state.catalog

    results = [
        ProductSummary(sku=product.sku, name=product.name, price=product.price,
                        category=product.category, color=product.color)
        for product in catalog
        if (category is None or product.category.lower() == category.lower())
        and (color is None or (product.color is not None and product.color.lower() == color.lower()))
    ]
    return ProductListResponse(results=results)
