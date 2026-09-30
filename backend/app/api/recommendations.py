"""HTTP API layer for shopper events and personalized recommendations.

Responsibility: HTTP concerns only (parsing, status codes) -- the
history/ranking logic is `RecommendationService`'s. Requires the
`X-API-Key` header at the router level and is rate-limited per client IP
like the other routes (these budgets are high because a storefront
sends all its shoppers' traffic from one IP -- see `app.config`).

No `from __future__ import annotations` in this module deliberately --
see the note in `api/search.py`.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request, status
from pydantic import Field

from app.auth import require_api_key
from app.config import (
    EVENTS_RATE_LIMIT,
    RECOMMEND_DEFAULT_LIMIT,
    RECOMMEND_MAX_LIMIT,
    RECOMMENDATIONS_RATE_LIMIT,
    SHOPPER_ID_PATTERN,
    SKU_PATTERN,
)
from app.rate_limit import limiter
from app.schemas.recommendation import (
    DeleteEventsResponse,
    EventBatch,
    EventsResponse,
    RecommendationResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["recommendations"], dependencies=[Depends(require_api_key)])

_MAX_EXCLUDED_SKUS = 20


@router.post("/events", response_model=EventsResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit(EVENTS_RATE_LIMIT)
async def record_events(request: Request, batch: EventBatch) -> EventsResponse:
    """Record shopper events (product views, add-to-carts, purchases).

    These are what `GET /recommendations` learns from. Events for SKUs
    that aren't (yet) in the catalog index are accepted and simply
    ignored until the product is indexed.

    Raises:
        HTTPException 401: Missing/invalid `X-API-Key`.
        HTTPException 422: Invalid shopper id / SKU / event type / timestamp,
            or an empty or oversized batch.
        HTTPException 429: Rate limit exceeded.
        HTTPException 500: Unexpected server-side failure.
    """
    try:
        recorded = request.app.state.recommendation_service.record_events(batch.events)
    except Exception as exc:  # noqa: BLE001 - last-resort safety net for the API layer
        logger.exception("Unexpected error while recording events")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal error while recording events."
        ) from exc
    return EventsResponse(recorded=recorded)


@router.get("/recommendations", response_model=RecommendationResponse)
@limiter.limit(RECOMMENDATIONS_RATE_LIMIT)
async def get_recommendations(
    request: Request,
    shopper_id: Annotated[str, Query(pattern=SHOPPER_ID_PATTERN, description="Opaque shopper id events were recorded under.")],
    limit: Annotated[int, Query(ge=1, le=RECOMMEND_MAX_LIMIT)] = RECOMMEND_DEFAULT_LIMIT,
    exclude_sku: Annotated[
        list[Annotated[str, Field(pattern=SKU_PATTERN)]],
        Query(max_length=_MAX_EXCLUDED_SKUS, description="SKU to never return; repeat the parameter for several."),
    ] = [],  # noqa: B006 - FastAPI copies query defaults; never mutated here
) -> RecommendationResponse:
    """Recommend products for one shopper based on their event history.

    Returns `strategy: "personalized"` when the shopper has usable
    history, `"popular"` for a cold-start shopper, or `"none"` when no
    events exist at all yet (see `RecommendationResponse`). Products the
    shopper already bought or added to a cart are never returned, nor
    are any `exclude_sku` values.

    Raises:
        HTTPException 401: Missing/invalid `X-API-Key`.
        HTTPException 422: Invalid `shopper_id`, `limit` or `exclude_sku`.
        HTTPException 429: Rate limit exceeded.
        HTTPException 500: Unexpected server-side failure.
    """
    try:
        strategy, results = request.app.state.recommendation_service.recommend(
            shopper_id=shopper_id, limit=limit, exclude_skus=exclude_sku
        )
    except Exception as exc:  # noqa: BLE001 - last-resort safety net for the API layer
        logger.exception("Unexpected error while computing recommendations")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error while computing recommendations.",
        ) from exc
    return RecommendationResponse(strategy=strategy, results=results)


@router.delete("/shoppers/{shopper_id}/events", response_model=DeleteEventsResponse)
@limiter.limit(EVENTS_RATE_LIMIT)
async def delete_shopper_events(
    request: Request,
    shopper_id: Annotated[str, Path(pattern=SHOPPER_ID_PATTERN)],
) -> DeleteEventsResponse:
    """Erase everything recorded for one shopper (e.g. a privacy/erasure request).

    Idempotent: deleting a shopper with no events returns `{"deleted": 0}`.

    Raises:
        HTTPException 401: Missing/invalid `X-API-Key`.
        HTTPException 422: Invalid `shopper_id`.
        HTTPException 429: Rate limit exceeded.
        HTTPException 500: Unexpected server-side failure.
    """
    try:
        deleted = request.app.state.recommendation_service.delete_shopper_events(shopper_id)
    except Exception as exc:  # noqa: BLE001 - last-resort safety net for the API layer
        logger.exception("Unexpected error while deleting events for a shopper")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal error while deleting events."
        ) from exc
    return DeleteEventsResponse(deleted=deleted)
