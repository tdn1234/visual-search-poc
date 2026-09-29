"""HTTP API layer for browsing and creating catalog products.

Responsibility: `GET /products` is exact-metadata filtering, run as a
SQL query against Postgres -- no image involved, no embedding math.
`POST /products` is the write counterpart: accepts a new product's
metadata plus a photo, and delegates to `IndexingService.add_product`
to embed the photo and persist both the catalog folder and the DB row,
*synchronously* -- the response only returns once the product is
fully stored. `POST /products/import` is the bulk counterpart for
that same write, used for e.g. a Magento catalog import: many
products in one request, each embedded *asynchronously* by a
background worker (`app/jobs.py` + `scripts/run_worker.py`) instead of
inline, so one slow/large batch can't tie up the API process or blow
past an HTTP client's timeout. None of these routes do their own
SQL/filesystem/queue work -- that's `services`'/`queue`'s job; this
module is HTTP concerns only (parsing, status codes).

Requires the `X-API-Key` header (`require_api_key`, applied at the
router level) and is rate-limited per client IP -- write routes get
tighter budgets than `GET /products` since they run CLIP inference
(`POST /products`, inline) or read many files into memory at once
(`POST /products/import`).

No `from __future__ import annotations` in this module deliberately --
see the note in `api/search.py` (slowapi's decorator + deferred string
annotations don't resolve cleanly through FastAPI's route introspection).
"""

import json
import logging
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile, status
from pydantic import ValidationError

from app.auth import require_api_key
from app.config import (
    BULK_IMPORT_RATE_LIMIT,
    CATALOG_DIR,
    CREATE_PRODUCT_RATE_LIMIT,
    MAX_BULK_IMPORT_ITEMS,
    PRODUCTS_RATE_LIMIT,
    SKU_PATTERN,
)
from app.queue import enqueue_import_job
from app.rate_limit import limiter
from app.schemas.search import BulkImportResponse, BulkProductItem, ProductListResponse, ProductSummary

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


@router.post("/products/import", response_model=BulkImportResponse, status_code=status.HTTP_202_ACCEPTED)
@limiter.limit(BULK_IMPORT_RATE_LIMIT)
async def bulk_import_products(
    request: Request,
    products: str = Form(
        ...,
        description=(
            "JSON array of product metadata, e.g. "
            '\'[{"sku":"bag-x","name":"...","price":9.99,"category":"Bags","color":"red"}]\''
            ". Must have exactly as many entries as `files`, in the same order -- "
            "products[0] is embedded from files[0], products[1] from files[1], etc."
        ),
    ),
    files: list[UploadFile] = File(..., description="Product photos (JPEG or PNG), same order as `products`."),
) -> BulkImportResponse:
    """Queue a batch of new products for background import (e.g. a Magento catalog export).

    Unlike `POST /products`, this does **not** wait for embedding/
    storage to finish -- it validates the batch's *shape* (JSON parses,
    counts match, fields are well-formed, images are a supported
    content type), enqueues one job per product onto a Redis-backed
    queue (`app.queue.enqueue_import_job`), and returns immediately.
    Actually storing each product (embedding the photo,
    writing `catalog/<sku>/`, inserting the DB row -- the same
    `IndexingService.add_product` `POST /products` calls synchronously)
    happens later, off the request, in a separate `worker` process
    (`scripts/run_worker.py`). If no worker is running, jobs still
    queue successfully but nothing processes them until one is.

    This is fire-and-forget by design: there is no status endpoint.
    Per-item outcomes -- including a duplicate `sku` or a corrupt image
    that only a worker can detect -- are in the worker's logs,
    correlated by this call's `batch_id`
    (`docker-compose logs worker | grep <batch_id>`).

    Args:
        request: The FastAPI request (used only for the rate limiter;
            no `app.state` service is needed here since this route
            only enqueues, it never touches Postgres or CLIP itself).
        products: A JSON-encoded array of per-product metadata
            (`BulkProductItem`: sku, name, price, category, color).
        files: The matching product photos, in the same order.

    Returns:
        A `BulkImportResponse` with a `batch_id` (for log correlation)
        and how many products were queued.

    Raises:
        HTTPException 400: If `products` isn't valid JSON, isn't an
            array, doesn't have exactly as many entries as `files`,
            any entry fails validation (bad `sku` format, non-positive
            `price`, missing field), the batch is empty, the batch
            exceeds `MAX_BULK_IMPORT_ITEMS`, or any file is
            missing/empty/an unsupported content type.
        HTTPException 401: If the request is missing a valid
            `X-API-Key` header.
        HTTPException 429: If the caller has exceeded `BULK_IMPORT_RATE_LIMIT`.
    """
    try:
        raw_items = json.loads(products)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"`products` is not valid JSON: {exc}") from exc

    if not isinstance(raw_items, list):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="`products` must be a JSON array.")

    if not raw_items:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No products given.")

    if len(raw_items) > MAX_BULK_IMPORT_ITEMS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Too many products in one batch ({len(raw_items)}); max is {MAX_BULK_IMPORT_ITEMS}.",
        )

    if len(raw_items) != len(files):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"`products` has {len(raw_items)} entries but {len(files)} files were uploaded; "
                "they must match 1:1, in order."
            ),
        )

    try:
        items = [BulkProductItem(**raw) for raw in raw_items]
    except (ValidationError, TypeError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid product metadata: {exc}") from exc

    # Validate + read every file *before* queuing anything, so a bad
    # file well into the batch (e.g. item 40 of 50) fails the whole
    # request instead of leaving the first 39 jobs queued with no way
    # to cancel them.
    validated: list[tuple[BulkProductItem, bytes, str]] = []
    for index, (item, file) in enumerate(zip(items, files)):
        if file.content_type not in _CONTENT_TYPE_TO_FILENAME:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"products[{index}] ('{item.sku}'): unsupported file type '{file.content_type}'. Use JPEG or PNG.",
            )
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"products[{index}] ('{item.sku}'): uploaded file is empty.",
            )
        validated.append((item, image_bytes, _CONTENT_TYPE_TO_FILENAME[file.content_type]))

    batch_id = uuid.uuid4().hex[:12]
    for index, (item, image_bytes, image_filename) in enumerate(validated):
        enqueue_import_job(
            batch_id=batch_id,
            item_index=index,
            sku=item.sku,
            name=item.name,
            price=item.price,
            category=item.category,
            color=item.color,
            image_bytes=image_bytes,
            image_filename=image_filename,
        )

    logger.info("Queued bulk import batch '%s': %d products", batch_id, len(validated))
    return BulkImportResponse(batch_id=batch_id, queued=len(validated))
