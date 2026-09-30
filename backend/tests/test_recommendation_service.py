"""RecommendationService: the taste-vector math (pure) and the SQL/parameters around it (Postgres faked).

The SQL itself was also run against a real pgvector Postgres when this
feature was built; these tests pin the parts that must not silently drift.
"""

import math

import numpy as np
import pytest

from app.config import EVENT_WEIGHTS, RECOMMEND_MAX_HISTORY, RECOMMEND_WINDOW_DAYS
from app.schemas.recommendation import EventItem
from app.services.recommendation_service import RecommendationService, build_taste_vector
from tests.fakes import FakePool, normalize_sql

E0, E1 = [1.0, 0.0], [0.0, 1.0]
HALF_LIFE = 10.0


def _taste(rows, **kwargs):
    return build_taste_vector(rows, weights=EVENT_WEIGHTS, half_life_days=HALF_LIFE, **kwargs)


# --- build_taste_vector -----------------------------------------------------


def test_no_history_means_no_taste_vector():
    assert _taste([]) is None


def test_single_event_gives_that_products_direction():
    taste = _taste([("view", 0.0, "a", [3.0, 0.0])])
    assert taste == pytest.approx([1.0, 0.0])


def test_result_is_unit_length():
    taste = _taste([("view", 1.0, "a", E0), ("purchase", 2.0, "b", E1), ("add_to_cart", 0.0, "c", [0.6, 0.8])])
    assert np.linalg.norm(taste) == pytest.approx(1.0)


def test_purchases_pull_harder_than_views():
    taste = _taste([("view", 0.0, "a", E0), ("purchase", 0.0, "b", E1)])
    # purchase weight 5 vs view weight 1 -> mostly toward E1
    assert taste[1] > taste[0]
    assert taste == pytest.approx(np.array([1.0, 5.0]) / math.sqrt(26))


def test_weights_follow_the_configured_ratio_view_lt_cart_lt_purchase():
    assert EVENT_WEIGHTS["view"] < EVENT_WEIGHTS["add_to_cart"] < EVENT_WEIGHTS["purchase"]


def test_older_events_count_less_by_exponential_half_life():
    # Equal type, but the E0 event is exactly one half-life old -> half the pull of the fresh E1 event.
    taste = _taste([("view", HALF_LIFE, "a", E0), ("view", 0.0, "b", E1)])
    assert taste == pytest.approx(np.array([0.5, 1.0]) / math.sqrt(1.25))


def test_a_very_old_event_barely_registers():
    taste = _taste([("purchase", 365.0, "old", E0), ("view", 0.0, "new", E1)])
    assert taste[1] > 0.999


def test_negative_age_from_clock_skew_is_not_boosted_above_a_fresh_event():
    skewed = _taste([("view", -3.0, "a", E0), ("view", 0.0, "b", E1)])
    assert skewed == pytest.approx(np.array([1.0, 1.0]) / math.sqrt(2))


def test_repeated_interest_in_the_same_direction_accumulates():
    one = _taste([("view", 0.0, "a", E0), ("view", 0.0, "b", E1)])
    three_views_of_e0 = _taste([("view", 0.0, "a", E0)] * 3 + [("view", 0.0, "b", E1)])
    assert three_views_of_e0[0] > one[0]


def test_unknown_event_types_are_ignored():
    assert _taste([("wishlist", 0.0, "a", E0)]) is None
    assert _taste([("wishlist", 0.0, "a", E0), ("view", 0.0, "b", E1)]) == pytest.approx([0.0, 1.0])


def test_exactly_cancelling_vectors_give_no_taste_instead_of_dividing_by_zero():
    assert _taste([("view", 0.0, "a", [1.0, 0.0]), ("view", 0.0, "b", [-1.0, 0.0])]) is None


# --- record_events / delete -------------------------------------------------


def _event(**overrides):
    data = {"shopper_id": "c1", "sku": "shoe-red", "event_type": "view"}
    data.update(overrides)
    return EventItem(**data)


def test_record_events_inserts_all_in_one_executemany_and_returns_the_count():
    pool = FakePool()
    events = [_event(), _event(sku="bag-x", event_type="purchase")]

    assert RecommendationService(pool).record_events(events) == 2

    (kind, sql, rows), = pool.calls
    assert kind == "executemany"
    assert "INSERT INTO shopper_events" in sql
    assert "COALESCE(%s::timestamptz, now())" in normalize_sql(sql)  # None -> DB clock
    assert rows == [("c1", "shoe-red", "view", None), ("c1", "bag-x", "purchase", None)]


def test_delete_returns_how_many_rows_the_database_removed():
    pool = FakePool(rowcount=7)

    assert RecommendationService(pool).delete_shopper_events("c1") == 7

    _, sql, params = pool.calls[0]
    assert normalize_sql(sql) == "DELETE FROM shopper_events WHERE shopper_id = %s"
    assert params == ("c1",)


# --- recommend --------------------------------------------------------------

# recommend() runs these queries in this order, so FakePool results are queued to match:
#   1 owned skus   2 history   3 nearest (personalized)  OR  3 popular (cold start)


def _personalized_pool(nearest_rows):
    owned = [("shoe-1",)]
    history = [("purchase", 1.0, "shoe-1", [1.0, 0.0]), ("view", 2.0, "shoe-2", [0.8, 0.6])]
    return FakePool(results=[owned, history, nearest_rows])


def test_personalized_result_uses_nearest_neighbours_of_the_taste_vector():
    pool = _personalized_pool([("shoe-3", "S3", 10.0, "Shoes", 0.1), ("shoe-4", "S4", 11.0, "Shoes", 0.25)])

    strategy, items = RecommendationService(pool).recommend("c1", limit=2)

    assert strategy == "personalized"
    assert [(i.sku, i.score) for i in items] == [("shoe-3", 0.9), ("shoe-4", 0.75)]  # 1 - distance
    _, sql, params = pool.calls[2]
    sql = normalize_sql(sql)
    assert "embedding <=> %s::vector" in sql and "sku <> ALL(%s::text[])" in sql
    taste_in_query, excluded, taste_again, limit = params
    assert len(taste_in_query) == 2 and taste_in_query == taste_again
    assert math.isclose(sum(x * x for x in taste_in_query), 1.0)  # a unit vector went to pgvector
    assert (excluded, limit) == (["shoe-1"], 2)


def test_owned_skus_and_caller_excludes_are_both_excluded_sorted_and_deduplicated():
    pool = _personalized_pool([])

    RecommendationService(pool).recommend("c1", limit=5, exclude_skus=["zzz", "shoe-1", "aaa"])

    assert pool.calls[2][2][1] == ["aaa", "shoe-1", "zzz"]


def test_owned_query_only_treats_purchases_and_carts_as_owned_within_the_window():
    pool = _personalized_pool([])

    RecommendationService(pool).recommend("c1", limit=5)

    _, sql, params = pool.calls[0]
    assert "event_type = ANY(%s::text[])" in normalize_sql(sql)
    assert params == ("c1", ["purchase", "add_to_cart"], RECOMMEND_WINDOW_DAYS)  # views stay recommendable


def test_history_query_is_bounded_and_scoped_to_the_shopper():
    pool = _personalized_pool([])

    RecommendationService(pool).recommend("c1", limit=5)

    _, sql, params = pool.calls[1]
    sql = normalize_sql(sql)
    assert "ORDER BY created_at DESC LIMIT %s" in sql  # newest events, cut off before the join
    assert "JOIN products p ON p.sku = e.sku" in sql  # unindexed skus drop out here
    assert params == ("c1", RECOMMEND_WINDOW_DAYS, RECOMMEND_MAX_HISTORY)


def test_score_is_clamped_to_zero_for_opposite_directions():
    pool = _personalized_pool([("far", "F", 1.0, "Shoes", 1.4)])  # cosine distance > 1
    _, items = RecommendationService(pool).recommend("c1", limit=1)
    assert items[0].score == 0.0


def test_no_usable_history_falls_back_to_popular_and_labels_it():
    pool = FakePool(results=[[], [], [("shoe-1", "S1", 10.0, "Shoes", 10.0), ("bag-1", "B1", 5.0, "Bags", 2.5)]])

    strategy, items = RecommendationService(pool).recommend("stranger", limit=5)

    assert strategy == "popular"
    assert [(i.sku, i.score) for i in items] == [("shoe-1", 1.0), ("bag-1", 0.25)]  # relative to the top item
    _, sql, params = pool.calls[2]
    assert "GROUP BY p.sku" in normalize_sql(sql)
    weights = params[:3]
    assert weights == (EVENT_WEIGHTS["purchase"], EVENT_WEIGHTS["add_to_cart"], EVENT_WEIGHTS["view"])


def test_popular_fallback_still_excludes_the_callers_skus():
    pool = FakePool(results=[[], [], []])

    RecommendationService(pool).recommend("stranger", limit=5, exclude_skus=["shoe-1"])

    assert pool.calls[2][2][4] == ["shoe-1"]


def test_strategy_is_none_when_there_is_nothing_to_recommend_at_all():
    pool = FakePool(results=[[], [], []])

    assert RecommendationService(pool).recommend("stranger", limit=5) == ("none", [])


def test_history_of_only_unindexed_products_is_treated_as_cold_start():
    # History query returns no rows because the JOIN to products dropped them all.
    pool = FakePool(results=[[("shoe-1",)], [], [("bag-1", "B", 1.0, "Bags", 3.0)]])

    strategy, _ = RecommendationService(pool).recommend("c1", limit=5)

    assert strategy == "popular"
