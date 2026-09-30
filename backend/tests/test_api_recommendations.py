"""POST /events, GET /recommendations, DELETE /shoppers/{id}/events: validation, status codes, hand-off."""

from datetime import datetime, timedelta, timezone

import pytest

from app.config import RECOMMEND_DEFAULT_LIMIT, RECOMMEND_MAX_LIMIT


def _events(client, headers, events):
    return client.post("/events", headers=headers, json={"events": events})


EVENT = {"shopper_id": "c42", "sku": "shoe-red", "event_type": "view"}


# --- POST /events -----------------------------------------------------------


def test_events_are_recorded_and_counted(client, auth_headers, services):
    response = _events(client, auth_headers, [EVENT, {**EVENT, "sku": "bag-x", "event_type": "purchase"}])

    assert response.status_code == 201
    assert response.json() == {"recorded": 2}
    (batch,) = services.recommendation.recorded_batches
    assert [(e.shopper_id, e.sku, e.event_type) for e in batch] == [
        ("c42", "shoe-red", "view"),
        ("c42", "bag-x", "purchase"),
    ]


def test_events_accept_a_backfilled_timestamp(client, auth_headers, services):
    when = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()

    assert _events(client, auth_headers, [{**EVENT, "occurred_at": when}]).status_code == 201
    assert services.recommendation.recorded_batches[0][0].occurred_at is not None


@pytest.mark.parametrize(
    "bad_events",
    [
        [],                                                     # empty batch
        [{**EVENT, "event_type": "click"}],                     # unknown type
        [{**EVENT, "shopper_id": "not valid!"}],                # bad shopper id
        [{**EVENT, "sku": "Bad_SKU"}],                          # bad sku
        [{**EVENT, "occurred_at": "2999-01-01T00:00:00Z"}],     # future
        [{"shopper_id": "c1", "sku": "a"}],                     # missing event_type
    ],
)
def test_invalid_events_are_422_and_nothing_is_recorded(client, auth_headers, services, bad_events):
    assert _events(client, auth_headers, bad_events).status_code == 422
    assert services.recommendation.recorded_batches == []


def test_one_bad_event_rejects_the_whole_batch(client, auth_headers, services):
    response = _events(client, auth_headers, [EVENT, {**EVENT, "event_type": "click"}])

    assert response.status_code == 422
    assert services.recommendation.recorded_batches == []  # all-or-nothing


def test_events_unexpected_failure_is_a_generic_500(client, auth_headers, services):
    services.recommendation.error = RuntimeError("password=hunter2 rejected")

    response = _events(client, auth_headers, [EVENT])

    assert response.status_code == 500
    assert "hunter2" not in response.text


# --- GET /recommendations ---------------------------------------------------


def test_recommendations_return_strategy_and_ranked_items(client, auth_headers, services):
    response = client.get("/recommendations", headers=auth_headers, params={"shopper_id": "c42"})

    assert response.status_code == 200
    assert response.json() == {
        "strategy": "personalized",
        "results": [{"sku": "shoe-blue", "name": "Blue Shoe", "price": 89.0, "category": "Shoes", "score": 0.83}],
    }


def test_recommendations_use_default_limit_and_no_excludes_when_not_given(client, auth_headers, services):
    client.get("/recommendations", headers=auth_headers, params={"shopper_id": "c42"})

    assert services.recommendation.recommend_calls == [
        {"shopper_id": "c42", "limit": RECOMMEND_DEFAULT_LIMIT, "exclude_skus": []}
    ]


def test_recommendations_pass_limit_and_repeated_exclude_sku_through(client, auth_headers, services):
    client.get(
        "/recommendations",
        headers=auth_headers,
        params={"shopper_id": "g1", "limit": 3, "exclude_sku": ["shoe-red", "bag-x"]},
    )

    assert services.recommendation.recommend_calls == [
        {"shopper_id": "g1", "limit": 3, "exclude_skus": ["shoe-red", "bag-x"]}
    ]


def test_cold_start_strategy_is_reported_as_is(client, auth_headers, services):
    services.recommendation.strategy = "popular"

    assert client.get("/recommendations", headers=auth_headers, params={"shopper_id": "new"}).json()["strategy"] == "popular"


def test_empty_result_is_a_normal_200(client, auth_headers, services):
    services.recommendation.strategy, services.recommendation.recommendations = "none", []

    response = client.get("/recommendations", headers=auth_headers, params={"shopper_id": "new"})

    assert response.status_code == 200
    assert response.json() == {"strategy": "none", "results": []}


@pytest.mark.parametrize(
    "params",
    [
        {},                                                    # shopper_id required
        {"shopper_id": "bad id"},
        {"shopper_id": "c1", "limit": 0},
        {"shopper_id": "c1", "limit": RECOMMEND_MAX_LIMIT + 1},
        {"shopper_id": "c1", "exclude_sku": "Bad_SKU"},
        {"shopper_id": "c1", "exclude_sku": [f"s{i}" for i in range(21)]},  # too many
    ],
)
def test_recommendations_validate_query_params(client, auth_headers, services, params):
    assert client.get("/recommendations", headers=auth_headers, params=params).status_code == 422
    assert services.recommendation.recommend_calls == []


def test_recommendations_unexpected_failure_is_a_generic_500(client, auth_headers, services):
    services.recommendation.error = RuntimeError("db at 10.0.0.5 refused")

    response = client.get("/recommendations", headers=auth_headers, params={"shopper_id": "c1"})

    assert response.status_code == 500
    assert "10.0.0.5" not in response.text


# --- DELETE /shoppers/{shopper_id}/events -----------------------------------


def test_delete_shopper_events_reports_how_many_were_erased(client, auth_headers, services):
    response = client.delete("/shoppers/c42/events", headers=auth_headers)

    assert response.status_code == 200
    assert response.json() == {"deleted": 4}
    assert services.recommendation.delete_calls == ["c42"]


def test_delete_rejects_a_malformed_shopper_id(client, auth_headers, services):
    assert client.delete("/shoppers/bad%20id/events", headers=auth_headers).status_code == 422
    assert services.recommendation.delete_calls == []


def test_delete_unexpected_failure_is_a_generic_500(client, auth_headers, services):
    services.recommendation.error = RuntimeError("boom")

    assert client.delete("/shoppers/c1/events", headers=auth_headers).status_code == 500
