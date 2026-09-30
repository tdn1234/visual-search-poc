"""Event/recommendation request validation."""

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from app.config import MAX_EVENTS_PER_REQUEST
from app.schemas.recommendation import EventBatch, EventItem


def _event(**overrides):
    data = {"shopper_id": "c42", "sku": "shoe-red", "event_type": "view"}
    data.update(overrides)
    return EventItem(**data)


@pytest.mark.parametrize("shopper_id", ["c42", "g9f2ab", "a-b_c", "X" * 64])
def test_valid_shopper_ids(shopper_id):
    assert _event(shopper_id=shopper_id).shopper_id == shopper_id


@pytest.mark.parametrize("shopper_id", ["", "X" * 65, "a b", "a/b", "../x", "me@example.com", "c42;drop"])
def test_shopper_ids_must_be_opaque_url_safe_tokens(shopper_id):
    with pytest.raises(ValidationError):
        _event(shopper_id=shopper_id)


@pytest.mark.parametrize("event_type", ["view", "add_to_cart", "purchase"])
def test_known_event_types(event_type):
    assert _event(event_type=event_type).event_type == event_type


@pytest.mark.parametrize("event_type", ["click", "VIEW", "", "wishlist"])
def test_unknown_event_types_are_rejected(event_type):
    with pytest.raises(ValidationError):
        _event(event_type=event_type)


def test_sku_uses_the_same_rule_as_the_catalog():
    with pytest.raises(ValidationError):
        _event(sku="Bad_SKU")


def test_occurred_at_is_optional():
    assert _event().occurred_at is None


def test_naive_timestamps_are_taken_as_utc():
    naive = datetime.now() - timedelta(days=1)
    assert _event(occurred_at=naive.replace(tzinfo=None)).occurred_at.tzinfo == timezone.utc


def test_past_timestamps_keep_their_timezone_for_backfilling_history():
    past = datetime.now(timezone.utc) - timedelta(days=200)
    assert _event(occurred_at=past).occurred_at == past


def test_timestamps_from_the_future_are_rejected_but_small_clock_skew_is_tolerated():
    now = datetime.now(timezone.utc)
    assert _event(occurred_at=now + timedelta(minutes=2)).occurred_at is not None
    with pytest.raises(ValidationError, match="future"):
        _event(occurred_at=now + timedelta(hours=1))


def test_batch_needs_at_least_one_event():
    with pytest.raises(ValidationError):
        EventBatch(events=[])


def test_batch_is_capped():
    event = {"shopper_id": "c1", "sku": "a", "event_type": "view"}
    assert len(EventBatch(events=[event] * MAX_EVENTS_PER_REQUEST).events) == MAX_EVENTS_PER_REQUEST
    with pytest.raises(ValidationError):
        EventBatch(events=[event] * (MAX_EVENTS_PER_REQUEST + 1))
