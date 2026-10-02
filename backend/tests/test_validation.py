"""Input hardening: upload size/format/pixel checks and text-field bounds, across all write routes."""

import io
import json

import pytest
from PIL import Image

from app import validation
from app.config import MAX_PRICE, NAME_MAX_LENGTH

FORM = {"sku": "shoe-purple", "name": "Purple Shoe", "price": "79.9", "category": "Shoes", "color": "purple"}


def _post_product(client, headers, image, content_type="image/png", form=None):
    return client.post("/products", headers=headers, data=form or FORM, files={"file": ("p", image, content_type)})


def _search(client, headers, image, content_type="image/png"):
    return client.post("/search", headers=headers, files={"file": ("q", image, content_type)})


@pytest.fixture
def small_limits(monkeypatch):
    monkeypatch.setattr(validation, "MAX_UPLOAD_BYTES", 100)
    monkeypatch.setattr(validation, "MAX_IMAGE_PIXELS", 50)


def test_oversized_upload_is_413_and_never_embedded(client, auth_headers, services, png_bytes, small_limits):
    padded = png_bytes + b"\0" * 200  # still starts as a valid PNG; size alone must reject it

    for response in (_search(client, auth_headers, padded), _post_product(client, auth_headers, padded)):
        assert response.status_code == 413
    assert services.embedding.bytes_calls == []
    assert services.indexing.calls == []


def test_image_with_too_many_pixels_is_rejected(client, auth_headers, services, monkeypatch, png_bytes):
    monkeypatch.setattr(validation, "MAX_IMAGE_PIXELS", 10)  # fixture image is 8x8 = 64

    response = _search(client, auth_headers, png_bytes)

    assert response.status_code == 400
    assert "pixel limit" in response.json()["detail"]
    assert services.embedding.bytes_calls == []


def test_declared_type_must_match_the_actual_format(client, auth_headers, services, jpeg_bytes):
    response = _search(client, auth_headers, jpeg_bytes, content_type="image/png")

    assert response.status_code == 400
    assert "does not match" in response.json()["detail"]
    assert services.embedding.bytes_calls == []


def test_non_image_bytes_with_an_image_content_type_are_rejected(client, auth_headers, services):
    for response in (
        _search(client, auth_headers, b"<?php system($_GET['c']); ?>"),
        _post_product(client, auth_headers, b"<svg onload=alert(1)/>"),
    ):
        assert response.status_code == 400
        assert "contents are not a supported image" in response.json()["detail"] or "not a valid image" in response.json()["detail"]
    assert services.indexing.calls == []


def test_truncated_image_is_rejected(client, auth_headers, services, png_bytes):
    response = _search(client, auth_headers, png_bytes[: len(png_bytes) // 2])

    assert response.status_code == 400
    assert services.embedding.bytes_calls == []


def test_stored_filename_follows_the_real_format(client, auth_headers, services, jpeg_bytes):
    _post_product(client, auth_headers, jpeg_bytes, content_type="image/jpeg")

    assert services.indexing.calls[0]["image_filename"] == "image.jpg"


def test_bulk_import_rejects_a_corrupt_file_before_queuing_anything(client, auth_headers, png_bytes, monkeypatch):
    queued = []
    monkeypatch.setattr("app.api.products.enqueue_import_job", lambda **kw: queued.append(kw))
    items = [{"sku": f"bag-{i}", "name": "Bag", "price": 1, "category": "Bags"} for i in range(2)]
    files = [("files", ("a.png", png_bytes, "image/png")), ("files", ("b.png", b"not an image", "image/png"))]

    response = client.post("/products/import", headers=auth_headers, data={"products": json.dumps(items)}, files=files)

    assert response.status_code == 400
    assert "products[1] ('bag-1')" in response.json()["detail"]
    assert queued == []


def test_bulk_import_enforces_a_total_size_cap(client, auth_headers, png_bytes, monkeypatch):
    queued = []
    monkeypatch.setattr("app.api.products.enqueue_import_job", lambda **kw: queued.append(kw))
    monkeypatch.setattr("app.api.products.MAX_BULK_IMPORT_TOTAL_BYTES", len(png_bytes) + 1)
    items = [{"sku": f"bag-{i}", "name": "Bag", "price": 1, "category": "Bags"} for i in range(2)]
    files = [("files", (f"{i}.png", png_bytes, "image/png")) for i in range(2)]

    response = client.post("/products/import", headers=auth_headers, data={"products": json.dumps(items)}, files=files)

    assert response.status_code == 413
    assert queued == []


def test_oversized_content_length_is_rejected_up_front(client, auth_headers, monkeypatch):
    from app import main

    # Rebuild the ASGI stack's check directly: the middleware only looks at the header.
    from app.middleware import BodySizeLimitMiddleware

    sent = []

    async def app(scope, receive, send):
        sent.append("reached app")

    async def send(message):
        sent.append(message)

    import asyncio

    scope = {"type": "http", "headers": [(b"content-length", b"1001")]}
    asyncio.run(BodySizeLimitMiddleware(app, max_bytes=1000)(scope, None, send))

    assert "reached app" not in sent
    assert sent[0]["status"] == 413
    assert main.app is not None


@pytest.mark.parametrize(
    "override",
    [
        {"name": "x" * (NAME_MAX_LENGTH + 1)},
        {"name": "line\nbreak"},
        {"name": "   "},
        {"category": "Shoes\x00"},
        {"color": "c" * 51},
        {"price": str(MAX_PRICE + 1)},
        {"price": "nan"},
        {"price": "inf"},
        {"price": "-1"},
    ],
)
def test_create_product_rejects_bad_metadata(client, auth_headers, services, png_bytes, override):
    response = _post_product(client, auth_headers, png_bytes, form={**FORM, **override})

    assert response.status_code == 422
    assert services.indexing.calls == []


def test_bulk_import_rejects_bad_metadata(client, auth_headers, png_bytes):
    item = {"sku": "bag-a", "name": "Bad\nName", "price": 1, "category": "Bags"}

    response = client.post(
        "/products/import",
        headers=auth_headers,
        data={"products": json.dumps([item])},
        files=[("files", ("a.png", png_bytes, "image/png"))],
    )

    assert response.status_code == 400
    assert "Invalid product metadata" in response.json()["detail"]


def test_list_products_rejects_overlong_filters(client, auth_headers, services):
    assert client.get("/products", headers=auth_headers, params={"category": "x" * 101}).status_code == 422
    assert client.get("/products", headers=auth_headers, params={"color": "x" * 51}).status_code == 422
