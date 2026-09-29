"""HTTP API layer for browsing and creating catalog products.

Responsibility: `GET /products` is exact-metadata filtering, run as a
SQL query against Postgres -- no image involved, no embedding math.
`POST /products` is the write counterpart: accepts a new product's
metadata plus a photo, and delegates to `IndexingService.add_product`
to embed the photo and persist both the catalog folder and the DB row.
Neither route does its own SQL/filesystem work -- that's `services`'
job; this module is HTTP concerns only (parsing, status codes).

Requires the `X-API-Key` header (`require_api_key`, applied at the
router level) and is rate-limited per client IP -- `POST /products`
gets a tighter budget than `GET /products` since it runs CLIP
inference and writes to disk, closer in cost to `POST /search` than
to a plain SQL filter.

No `from __future__ import annotations` in this module deliberately --
see the note in `api/search.py` (slowapi's decorator + deferred string
annotations don't resolve cleanly through FastAPI's route introspection).
"""

import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile, status

from app.auth import require_api_key
from app.config import CATALOG_DIR, CREATE_PRODUCT_RATE_LIMIT, PRODUCTS_RATE_LIMIT, SKU_PATTERN
from app.rate_limit import limiter
from app.schemas.search import ProductListResponse, ProductSummary

logger = logging.getLogger(__name__)

router = APIRouter(tags=["products"], dependencies=[Depends(require_api_key)])

# Restricted to formats IndexingService.build_index's catalog scan also
# recognizes (SUPPORTED_IMAGE_NAMES has no ".webp") -- otherwise a
# product added here would silently vanish from a later full reindex.
_CONTENT_TYPE_TO_FILENAME = {
    "image/jpeg": "image.jpg",
    "image/png": "image.png",
}


@router.get("/products", response_model=ProductListResponse)
@limiter.limit(PRODUCTS_RATE_LIMIT)
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
        request: The FastAPI request, used to reach the
            `ProductQueryService` stored on `app.state` (set up once
            at startup).
        category: If given, only return products in this category.
        color: If given, only return products with this color.

    Returns:
        A `ProductListResponse` listing every matching product (no
        embeddings, no ranking/score -- this is a plain filter).

    Raises:
        HTTPException 401: If the request is missing a valid
            `X-API-Key` header.
        HTTPException 429: If the caller has exceeded `PRODUCTS_RATE_LIMIT`.
    """
    product_query_service = request.app.state.product_query_service
    results = product_query_service.list_products(category=category, color=color)
    return ProductListResponse(results=results)


@router.post("/products", response_model=ProductSummary, status_code=status.HTTP_201_CREATED)
@limiter.limit(CREATE_PRODUCT_RATE_LIMIT)
async def create_product(
    request: Request,
    sku: str = Form(
        ...,
        pattern=SKU_PATTERN,
        description="Unique product identifier, e.g. 'shoe-purple'. Lowercase letters, digits, and hyphens only.",
    ),
    name: str = Form(..., description="Human-readable product name."),
    price: float = Form(..., gt=0, description="Product price."),
    category: str = Form(..., description="Product category, e.g. 'Shoes'."),
    color: str | None = Form(None, description="Dominant product color, e.g. 'purple'. Optional."),
    file: UploadFile = File(..., description="Product photo (JPEG or PNG)."),
) -> ProductSummary:
    """Add a new product: store its metadata and a CLIP embedding of its photo.

    Pipeline: validate the upload -> read its bytes -> hand everything
    to `IndexingService.add_product`, which writes `catalog/<sku>/`
    (image + metadata.json, so a later full reindex doesn't drop this
    product) and inserts one row into the `products` table -- using
    the exact same `EmbeddingService` transformation `POST /search`
    uses for query images, so this product is comparable against
    existing embeddings (and through any active color adapter)
    immediately, no separate reindex step required.

    Args:
        request: The FastAPI request, used to reach the
            `IndexingService` stored on `app.state`.
        sku: Unique product identifier (also becomes a literal
            `catalog/<sku>/` folder name -- `SKU_PATTERN` rejects
            anything that isn't a safe, simple slug).
        name: Human-readable product name.
        price: Product price (must be > 0).
        category: Product category, e.g. "Shoes".
        color: Optional dominant color.
        file: The product photo, sent as multipart/form-data.

    Returns:
        A `ProductSummary` for the newly created product.

    Raises:
        HTTPException 400: If the uploaded file is missing/empty/not
            a supported image type (or not a valid image at all).
        HTTPException 401: If the request is missing a valid
            `X-API-Key` header.
        HTTPException 409: If a product with this `sku` already exists.
        HTTPException 429: If the caller has exceeded `CREATE_PRODUCT_RATE_LIMIT`.
        HTTPException 500: For any unexpected server-side failure.
    """
    if file.content_type not in _CONTENT_TYPE_TO_FILENAME:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{file.content_type}'. Use JPEG or PNG.",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

    indexing_service = request.app.state.indexing_service

    try:
        record = indexing_service.add_product(
            catalog_dir=CATALOG_DIR,
            sku=sku,
            name=name,
            price=price,
            category=category,
            color=color,
            image_bytes=image_bytes,
            image_filename=_CONTENT_TYPE_TO_FILENAME[file.content_type],
        )
    except FileExistsError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - last-resort safety net for the API layer
        logger.exception("Unexpected error while creating product '%s'", sku)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error while creating the product.",
        ) from exc

    return ProductSummary(
        sku=record.sku, name=record.name, price=record.price, category=record.category, color=record.color
    )
