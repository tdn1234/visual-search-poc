"""Pydantic schemas for shopper events and personalized recommendations."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.config import MAX_EVENTS_PER_REQUEST, SHOPPER_ID_PATTERN, SKU_PATTERN

EventType = Literal["view", "add_to_cart", "purchase"]

# Tolerate small clock skew between the storefront and this service, but
# not events "from the future" (they'd sort ahead of real activity).
_MAX_FUTURE_SKEW = timedelta(minutes=5)


class EventItem(BaseModel):
    """One thing a shopper did to one product."""

    shopper_id: str = Field(..., pattern=SHOPPER_ID_PATTERN, description="Opaque shopper id, e.g. 'c42' or 'g9f2ab'.")
    sku: str = Field(..., pattern=SKU_PATTERN, description="The product's SKU as indexed here (not the storefront's raw SKU).")
    event_type: EventType
    occurred_at: datetime | None = Field(
        None,
        description="When it happened (ISO 8601). Omit for 'now'; set it to backfill historical orders.",
    )

    @field_validator("occurred_at")
    @classmethod
    def _normalize_and_bound(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        if value > datetime.now(timezone.utc) + _MAX_FUTURE_SKEW:
            raise ValueError("occurred_at must not be in the future")
        return value


class EventBatch(BaseModel):
    """Body of `POST /events`."""

    events: list[EventItem] = Field(..., min_length=1, max_length=MAX_EVENTS_PER_REQUEST)


class EventsResponse(BaseModel):
    recorded: int


class DeleteEventsResponse(BaseModel):
    deleted: int


class RecommendationItem(BaseModel):
    """A recommended product. `score` is cosine similarity to the shopper's taste
    (`personalized`) or popularity relative to the top item (`popular`); both are
    in [0, 1] and only comparable within one response."""

    sku: str
    name: str
    price: float
    category: str
    score: float


class RecommendationResponse(BaseModel):
    """`strategy` says how the list was produced:

    - `personalized`: built from this shopper's own history.
    - `popular`: the shopper has no usable history yet (cold start), so these
      are the most-engaged-with products overall.
    - `none`: no history for this shopper and no events at all yet.
    """

    strategy: Literal["personalized", "popular", "none"]
    results: list[RecommendationItem]
