"""HTTP API layer for visual search.

Responsibility: parse/validate the HTTP request, delegate all real
work to services (embedding + similarity), and shape the response.
No business logic (CLIP calls, cosine similarity math) lives here.
"""

from __future__ import annotations

import io
import logging

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile, status
from PIL import Image

from app.config import TOP_K_RESULTS
from app.schemas.search import SearchResponse
from app.services.catalog_filter import filter_by_category_or_color

logger = logging.getLogger(__name__)

router = APIRouter(tags=["search"])

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


@router.post("/search", response_model=SearchResponse)
async def search_by_image(
    request: Request,
    file: UploadFile = File(...),
    match_category: bool = Query(
        False, description="Only return items sharing the uploaded image's own category (auto-detected)."
    ),
    match_color: bool = Query(
        False, description="Only return items sharing the uploaded image's own color (auto-detected)."
    ),
) -> SearchResponse:
    """Find the top-K most visually similar products to an uploaded image.

    Pipeline: uploaded image bytes -> CLIP embedding -> (optional
    category/color classification + filter) -> cosine similarity
    against the indexed catalog -> top-K ranked results.

    `match_category`/`match_color` don't take a value -- checking one
    means "classify the *uploaded image itself* against the
    category/color labels present in the catalog (zero-shot, via
    `ClipModel.classify`), then only return items sharing whichever
    label wins." Checking both is OR: a result is kept if it matches
    the image's predicted category, or its predicted color, or both.

    Args:
        request: The FastAPI request, used to reach services stored on
            `app.state` (set up once at startup in `main.py`).
        file: The uploaded image, sent as multipart/form-data.
        match_category: If true, restrict results to the uploaded
            image's auto-detected category.
        match_color: If true, restrict results to the uploaded image's
            auto-detected color.

    Returns:
        A `SearchResponse` containing up to `TOP_K_RESULTS` matches
        (fewer if filtering narrowed the candidate pool), sorted by
        descending similarity score. Empty if filtering left no
        candidates.

    Raises:
        HTTPException 400: If the uploaded file is missing/empty/not
            a supported image type.
        HTTPException 503: If the product catalog has not been
            indexed yet (embeddings.json is empty/missing).
        HTTPException 500: For any unexpected server-side failure.
    """
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{file.content_type}'. Use JPEG, PNG, or WEBP.",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

    embedding_service = request.app.state.embedding_service
    similarity_service = request.app.state.similarity_service
    attribute_classifier_service = request.app.state.attribute_classifier_service
    catalog = request.app.state.catalog

    if not catalog:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Product catalog is empty. Run 'python scripts/build_embeddings.py' "
            "to index products before searching.",
        )

    try:
        query_embedding = embedding_service.embed_image_bytes(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    candidates = catalog
    if match_category or match_color:
        image = Image.open(io.BytesIO(image_bytes))  # already validated by embed_image_bytes above
        image.load()

        category_labels = sorted({p.category for p in catalog}) if match_category else None
        color_labels = sorted({p.color for p in catalog if p.color}) if match_color else None
        predicted_category, predicted_color = attribute_classifier_service.classify(
            image, category_labels=category_labels, color_labels=color_labels
        )

        candidates = filter_by_category_or_color(catalog, category=predicted_category, color=predicted_color)
        if not candidates:
            return SearchResponse(results=[])

    try:
        results = similarity_service.rank_products(
            query_embedding=query_embedding,
            catalog=candidates,
            top_k=TOP_K_RESULTS,
        )
    except Exception as exc:  # noqa: BLE001 - last-resort safety net for the API layer
        logger.exception("Unexpected error while ranking products")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error while searching for similar products.",
        ) from exc

    return SearchResponse(results=results)
