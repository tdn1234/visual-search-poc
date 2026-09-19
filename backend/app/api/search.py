"""HTTP API layer for visual search.

Responsibility: parse/validate the HTTP request, delegate all real
work to services (embedding + similarity), and shape the response.
No business logic (CLIP calls, cosine similarity math) lives here.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status

from app.config import TOP_K_RESULTS
from app.schemas.search import SearchResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["search"])

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


@router.post("/search", response_model=SearchResponse)
async def search_by_image(request: Request, file: UploadFile = File(...)) -> SearchResponse:
    """Find the top-K most visually similar products to an uploaded image.

    Pipeline: uploaded image bytes -> CLIP embedding -> cosine
    similarity against the indexed catalog -> top-K ranked results.

    Args:
        request: The FastAPI request, used to reach services stored on
            `app.state` (set up once at startup in `main.py`).
        file: The uploaded image, sent as multipart/form-data.

    Returns:
        A `SearchResponse` containing up to `TOP_K_RESULTS` matches,
        sorted by descending similarity score.

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
    catalog = request.app.state.catalog

    try:
        query_embedding = embedding_service.embed_image_bytes(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    try:
        results = similarity_service.rank_products(
            query_embedding=query_embedding,
            catalog=catalog,
            top_k=TOP_K_RESULTS,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - last-resort safety net for the API layer
        logger.exception("Unexpected error while ranking products")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error while searching for similar products.",
        ) from exc

    return SearchResponse(results=results)
