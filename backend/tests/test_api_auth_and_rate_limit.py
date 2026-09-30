"""API-key auth and per-IP rate limiting, exercised through the real app wiring."""

import pytest


@pytest.mark.parametrize(
    ("method", "path"),
    [("get", "/products"), ("post", "/products"), ("post", "/products/import"), ("post", "/search")],
)
def test_every_protected_route_rejects_a_missing_key(client, method, path):
    assert getattr(client, method)(path).status_code == 401


def test_wrong_key_is_rejected_with_a_helpful_message(client):
    response = client.get("/products", headers={"X-API-Key": "wrong"})

    assert response.status_code == 401
    assert "X-API-Key" in response.json()["detail"]


def test_correct_key_is_accepted(client, auth_headers):
    assert client.get("/products", headers=auth_headers).status_code == 200


def test_auth_is_checked_before_any_work_is_done(client, services, png_bytes):
    client.post("/search", files={"file": ("q.png", png_bytes, "image/png")})

    assert services.embedding.bytes_calls == []  # unauthenticated callers never reach CLIP


def test_search_is_rate_limited_per_client_with_429(client, auth_headers, png_bytes):
    """conftest sets SEARCH_RATE_LIMIT=5/minute."""
    upload = {"file": ("q.png", png_bytes, "image/png")}

    statuses = [client.post("/search", headers=auth_headers, files=upload).status_code for _ in range(6)]

    assert statuses == [200, 200, 200, 200, 200, 429]
