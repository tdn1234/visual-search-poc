"""Recommendation service.

Responsibility: everything behind `POST /events`, `DELETE
/shoppers/{id}/events` and `GET /recommendations` -- recording what
shoppers do, and turning one shopper's history into product
suggestions. Like `ProductQueryService`, all ranking happens in
Postgres/pgvector; this class only builds the query vector and shapes
the SQL.

How a recommendation is made (see docs/architecture.md, "Personalized
recommendations"):

1. Load the shopper's most recent events that point at indexed products.
2. Weight each one -- by event type (a purchase outweighs a view) and by
   age (exponential decay with a configurable half-life) -- and average
   the products' embeddings into one **taste vector**.
3. Ask pgvector for the products nearest that vector, excluding anything
   the shopper already bought or put in a cart (plus caller-supplied
   SKUs, e.g. the product page currently open).

CLIP embeddings are L2-normalized and live in one shared space, so a
weighted average of them is a meaningful "centre" of what someone
looks at, and cosine similarity to it is a sensible relevance score.

A shopper with no usable history (cold start) gets the most-engaged-with
products overall instead, labelled `popular` so the caller can tell.
"""

from __future__ import annotations

import logging
import time
from typing import Sequence

import numpy as np

from app.config import (
    EVENT_WEIGHTS,
    RECOMMEND_HALF_LIFE_DAYS,
    RECOMMEND_MAX_HISTORY,
    RECOMMEND_WINDOW_DAYS,
)
from app.schemas.recommendation import EventItem, RecommendationItem

logger = logging.getLogger(__name__)

# A history row as loaded from SQL: (event_type, age_in_days, sku, embedding).
HistoryRow = tuple[str, float, str, Sequence[float]]

# Events that mean "already has it / about to have it" -- never recommend those again.
_OWNED_EVENT_TYPES = ("purchase", "add_to_cart")


def build_taste_vector(
    history: Sequence[HistoryRow],
    weights: dict[str, float] = EVENT_WEIGHTS,
    half_life_days: float = RECOMMEND_HALF_LIFE_DAYS,
) -> np.ndarray | None:
    """Collapse a shopper's history into one unit-length "taste" embedding.

    Each event contributes `weights[event_type] * 0.5 ** (age_days / half_life_days)`
    times its product's embedding; the sum is L2-normalized so it can be
    compared with cosine distance like any other embedding here.

    Args:
        history: Rows of `(event_type, age_days, sku, embedding)`.
        weights: How much each event type counts.
        half_life_days: Age at which an event's weight has halved.

    Returns:
        The normalized taste vector, or `None` if there is nothing to
        average (empty history, unknown event types only, or weights that
        cancel to a zero vector).
    """
    total: np.ndarray | None = None
    for event_type, age_days, _sku, embedding in history:
        base_weight = weights.get(event_type)
        if base_weight is None:
            continue
        # Clock skew can make an event look slightly negative-aged; never boost it.
        decay = 0.5 ** (max(age_days, 0.0) / half_life_days)
        contribution = base_weight * decay * np.asarray(embedding, dtype=np.float64)
        total = contribution if total is None else total + contribution

    if total is None:
        return None
    norm = float(np.linalg.norm(total))
    if norm == 0.0:
        return None
    return total / norm


class RecommendationService:
    """Records shopper events and computes personalized recommendations."""

    def __init__(self, db_pool) -> None:
        """Store a reference to an already-open pool (schema already ensured to exist)."""
        self._db_pool = db_pool

    # --- writes ---------------------------------------------------------

    def record_events(self, events: Sequence[EventItem]) -> int:
        """Insert events in one round trip. `occurred_at=None` means "now" (DB clock).

        Returns:
            How many events were recorded.
        """
        rows = [(e.shopper_id, e.sku, e.event_type, e.occurred_at) for e in events]
        with self._db_pool.connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO shopper_events (shopper_id, sku, event_type, created_at)
                    VALUES (%s, %s, %s, COALESCE(%s::timestamptz, now()))
                    """,
                    rows,
                )
        return len(rows)

    def delete_shopper_events(self, shopper_id: str) -> int:
        """Erase every event for one shopper (privacy/GDPR erasure). Returns rows deleted."""
        with self._db_pool.connection() as conn:
            result = conn.execute("DELETE FROM shopper_events WHERE shopper_id = %s", (shopper_id,))
            return result.rowcount

    # --- reads ----------------------------------------------------------

    def recommend(
        self, shopper_id: str, limit: int, exclude_skus: Sequence[str] = ()
    ) -> tuple[str, list[RecommendationItem]]:
        """Recommend up to `limit` products for one shopper.

        Args:
            shopper_id: The opaque shopper id events were recorded under.
            limit: Maximum number of products to return.
            exclude_skus: Extra SKUs never to return (e.g. the product page being viewed).

        Returns:
            `(strategy, items)` where strategy is `"personalized"`,
            `"popular"` (cold start) or `"none"` (nothing to recommend yet).
        """
        start_time = time.perf_counter()

        owned = set(exclude_skus) | self._owned_skus(shopper_id)
        taste = build_taste_vector(self._load_history(shopper_id))

        if taste is not None:
            items = self._nearest_to(taste, owned, limit)
            strategy = "personalized"
        else:
            items = self._popular(owned, limit)
            strategy = "popular" if items else "none"

        logger.debug(
            "recommend(shopper=%s, limit=%d) -> %s, %d items, %.1fms",
            shopper_id, limit, strategy, len(items), (time.perf_counter() - start_time) * 1000,
        )
        return strategy, items

    def _owned_skus(self, shopper_id: str) -> set[str]:
        with self._db_pool.connection() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT sku FROM shopper_events
                WHERE shopper_id = %s
                  AND event_type = ANY(%s::text[])
                  AND created_at > now() - make_interval(days => %s)
                """,
                (shopper_id, list(_OWNED_EVENT_TYPES), RECOMMEND_WINDOW_DAYS),
            ).fetchall()
        return {row[0] for row in rows}

    def _load_history(self, shopper_id: str) -> list[HistoryRow]:
        """The shopper's newest events that point at indexed products, with their embeddings.

        The inner LIMIT applies to *events* (newest first) before the join, so
        a shopper's history is bounded by `RECOMMEND_MAX_HISTORY` regardless of size.
        """
        with self._db_pool.connection() as conn:
            rows = conn.execute(
                """
                SELECT e.event_type,
                       EXTRACT(EPOCH FROM (now() - e.created_at)) / 86400.0 AS age_days,
                       p.sku,
                       p.embedding
                FROM (
                    SELECT event_type, sku, created_at
                    FROM shopper_events
                    WHERE shopper_id = %s
                      AND created_at > now() - make_interval(days => %s)
                    ORDER BY created_at DESC
                    LIMIT %s
                ) e
                JOIN products p ON p.sku = e.sku
                """,
                (shopper_id, RECOMMEND_WINDOW_DAYS, RECOMMEND_MAX_HISTORY),
            ).fetchall()
        return [(row[0], float(row[1]), row[2], row[3]) for row in rows]

    def _nearest_to(self, taste: np.ndarray, excluded_skus: set[str], limit: int) -> list[RecommendationItem]:
        # `::vector` / `::text[]` casts for the same reason as in
        # ProductQueryService.search_similar: a bare Python list binds as
        # double precision[] / an untyped literal, which has no matching operator.
        taste_list = taste.tolist()
        with self._db_pool.connection() as conn:
            rows = conn.execute(
                """
                SELECT sku, name, price, category, embedding <=> %s::vector AS distance
                FROM products
                WHERE sku <> ALL(%s::text[])
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (taste_list, sorted(excluded_skus), taste_list, limit),
            ).fetchall()
        return [
            RecommendationItem(
                sku=row[0], name=row[1], price=row[2], category=row[3],
                # Cosine distance -> similarity, clamped to the documented [0, 1] range.
                score=round(max(0.0, 1.0 - float(row[4])), 4),
            )
            for row in rows
        ]

    def _popular(self, excluded_skus: set[str], limit: int) -> list[RecommendationItem]:
        """Cold-start fallback: products with the most (weighted) recent engagement overall."""
        with self._db_pool.connection() as conn:
            rows = conn.execute(
                """
                SELECT p.sku, p.name, p.price, p.category,
                       SUM(CASE e.event_type
                               WHEN 'purchase' THEN %s
                               WHEN 'add_to_cart' THEN %s
                               ELSE %s
                           END) AS popularity
                FROM shopper_events e
                JOIN products p ON p.sku = e.sku
                WHERE e.created_at > now() - make_interval(days => %s)
                  AND p.sku <> ALL(%s::text[])
                GROUP BY p.sku, p.name, p.price, p.category
                ORDER BY popularity DESC, p.sku
                LIMIT %s
                """,
                (
                    EVENT_WEIGHTS["purchase"],
                    EVENT_WEIGHTS["add_to_cart"],
                    EVENT_WEIGHTS["view"],
                    RECOMMEND_WINDOW_DAYS,
                    sorted(excluded_skus),
                    limit,
                ),
            ).fetchall()
        if not rows:
            return []
        top = float(rows[0][4]) or 1.0
        return [
            RecommendationItem(
                sku=row[0], name=row[1], price=row[2], category=row[3],
                score=round(float(row[4]) / top, 4),  # relative to the most popular item
            )
            for row in rows
        ]
