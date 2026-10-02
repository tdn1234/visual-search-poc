"""GET /products, POST /products, POST /products/import: validation, status codes, hand-off to services/queue."""

import json

import pytest

from app.api import products as products_module
from app.config import CATALOG_DIR

FORM = {"sku": "shoe-purple", "name": "Purple Shoe", "price": "79.9", "category": "Shoes", "color": "purple"}


# --- GET /products ----------------------------------------------------------


def test_list_products_passes_filters_through(client, auth_headers, services):
    response = client.get("/products", headers=auth_headers, params={"category": "Shoes", "color": "red"})

    assert response.status_code == 200
    assert services.product_query.list_calls == [{"category": "Shoes", "color": "red"}]
    assert response.json()["results"][0] == {
        "sku": "shoe-red", "name": "Red Shoe", "price": 99.0, "category": "Shoes", "color": "red",
    }


def test_list_products_without_filters_passes_none(client, auth_headers, services):
    client.get("/products", headers=auth_headers)
    assert services.product_query.list_calls == [{"category": None, "color": None}]


# --- POST /products ---------------------------------------------------------


def _create(client, headers, png_bytes, *, form=None, content_type="image/png", content=None):
    body = png_bytes if content is None else content
    return client.post("/products", headers=headers, data=form or FORM, files={"file": ("p", body, content_type)})


def test_create_product_returns_201_and_hands_everything_to_the_indexing_service(
    client, auth_headers, services, png_bytes
):
    response = _create(client, auth_headers, png_bytes)

    assert response.status_code == 201
    assert response.json() == {
        "sku": "shoe-purple", "name": "Purple Shoe", "price": 79.9, "category": "Shoes", "color": "purple",
    }
    (call,) = services.indexing.calls
    assert call == {
        "catalog_dir": CATALOG_DIR,
        "sku": "shoe-purple",
        "name": "Purple Shoe",
        "price": 79.9,
        "category": "Shoes",
        "color": "purple",
        "image_bytes": png_bytes,
        "image_filename": "image.png",
    }


def test_create_product_stores_jpeg_uploads_as_image_jpg(client, auth_headers, services, jpeg_bytes):
    _create(client, auth_headers, None, content=jpeg_bytes, content_type="image/jpeg")
    assert services.indexing.calls[0]["image_filename"] == "image.jpg"


def test_create_product_color_is_optional(client, auth_headers, services, png_bytes):
    form = {k: v for k, v in FORM.items() if k != "color"}

    response = _create(client, auth_headers, png_bytes, form=form)

    assert response.status_code == 201
    assert services.indexing.calls[0]["color"] is None


@pytest.mark.parametrize("content_type", ["image/webp", "image/gif", "application/pdf"])
def test_create_product_rejects_formats_a_later_reindex_would_not_recognize(
    client, auth_headers, services, content_type
):
    response = _create(client, auth_headers, None, content_type=content_type, content=b"data")

    assert response.status_code == 400
    assert "Use JPEG or PNG" in response.json()["detail"]
    assert services.indexing.calls == []


def test_create_product_rejects_an_empty_file(client, auth_headers, services):
    response = _create(client, auth_headers, None, content=b"")

    assert response.status_code == 400
    assert services.indexing.calls == []


@pytest.mark.parametrize(
    "bad_form",
    [
        {**FORM, "sku": "Purple_Shoe"},   # violates SKU_PATTERN
        {**FORM, "sku": "../escape"},     # path traversal attempt
        {**FORM, "price": "0"},           # must be > 0
        {**FORM, "price": "-5"},
        {**FORM, "price": "abc"},
        {k: v for k, v in FORM.items() if k != "name"},  # required field missing
    ],
)
def test_create_product_validates_form_fields_with_422(client, auth_headers, services, png_bytes, bad_form):
    response = _create(client, auth_headers, png_bytes, form=bad_form)

    assert response.status_code == 422
    assert services.indexing.calls == []


def test_create_product_duplicate_sku_is_409(client, auth_headers, services, png_bytes):
    services.indexing.error = FileExistsError("Product 'shoe-purple' already exists.")

    response = _create(client, auth_headers, png_bytes)

    assert response.status_code == 409
    assert "already exists" in response.json()["detail"]


def test_create_product_undecodable_image_is_400(client, auth_headers, services, png_bytes):
    services.indexing.error = ValueError("Uploaded file is not a valid image (jpg/png/webp).")

    response = _create(client, auth_headers, png_bytes)

    assert response.status_code == 400


def test_create_product_unexpected_failure_is_500_without_leaking_internals(
    client, auth_headers, services, png_bytes
):
    services.indexing.error = RuntimeError("password=hunter2 rejected by db")

    response = _create(client, auth_headers, png_bytes)

    assert response.status_code == 500
    assert "hunter2" not in response.text


# --- POST /products/import --------------------------------------------------


@pytest.fixture
def enqueued(monkeypatch):
    """Replace the RabbitMQ producer; returns the list of kwargs each job was queued with."""
    jobs: list[dict] = []
    monkeypatch.setattr(products_module, "enqueue_import_job", lambda **kwargs: jobs.append(kwargs))
    return jobs


def _item(sku, **overrides):
    return {"sku": sku, "name": sku.title(), "price": 10.0, "category": "Bags", "color": "red", **overrides}


def _import(client, headers, items, files):
    return client.post(
        "/products/import",
        headers=headers,
        data={"products": items if isinstance(items, str) else json.dumps(items)},
        files=files,
    )


def _png_files(png_bytes, count):
    return [("files", (f"f{i}.png", png_bytes, "image/png")) for i in range(count)]


def test_bulk_import_queues_one_job_per_product_in_order_and_returns_202(
    client, auth_headers, enqueued, png_bytes, jpeg_bytes
):
    files = [
        ("files", ("a.png", png_bytes, "image/png")),
        ("files", ("b.jpg", jpeg_bytes, "image/jpeg")),
    ]

    response = _import(client, auth_headers, [_item("bag-a"), _item("bag-b", color=None)], files)

    assert response.status_code == 202
    body = response.json()
    assert body["queued"] == 2
    assert len(body["batch_id"]) == 12

    assert [(j["item_index"], j["sku"], j["image_filename"]) for j in enqueued] == [
        (0, "bag-a", "image.png"),
        (1, "bag-b", "image.jpg"),
    ]
    assert enqueued[0]["image_bytes"] == png_bytes  # products[i] is paired with files[i]
    assert enqueued[1]["image_bytes"] == jpeg_bytes
    assert enqueued[1]["color"] is None
    assert {j["batch_id"] for j in enqueued} == {body["batch_id"]}  # one shared id for log correlation


@pytest.mark.parametrize(
    ("products", "expected_detail"),
    [
        ("{not json", "not valid JSON"),
        ('{"sku": "a"}', "must be a JSON array"),
        ("[]", "No products given"),
    ],
)
def test_bulk_import_rejects_malformed_product_lists(client, auth_headers, enqueued, png_bytes, products, expected_detail):
    response = _import(client, auth_headers, products, _png_files(png_bytes, 1))

    assert response.status_code == 400
    assert expected_detail in response.json()["detail"]
    assert enqueued == []


def test_bulk_import_rejects_products_and_files_that_do_not_match_one_to_one(client, auth_headers, enqueued, png_bytes):
    response = _import(client, auth_headers, [_item("bag-a"), _item("bag-b")], _png_files(png_bytes, 1))

    assert response.status_code == 400
    assert "must match 1:1" in response.json()["detail"]
    assert enqueued == []


def test_bulk_import_enforces_the_batch_size_cap(client, auth_headers, enqueued, png_bytes, monkeypatch):
    monkeypatch.setattr(products_module, "MAX_BULK_IMPORT_ITEMS", 2)

    response = _import(client, auth_headers, [_item(f"bag-{i}") for i in range(3)], _png_files(png_bytes, 3))

    assert response.status_code == 400
    assert "max is 2" in response.json()["detail"]
    assert enqueued == []


@pytest.mark.parametrize(
    "bad_item",
    [
        _item("Bad_SKU"),
        _item("bag-a", price=0),
        {"sku": "bag-a", "name": "x", "price": 1.0},  # missing category
    ],
)
def test_bulk_import_rejects_invalid_product_metadata(client, auth_headers, enqueued, png_bytes, bad_item):
    response = _import(client, auth_headers, [bad_item], _png_files(png_bytes, 1))

    assert response.status_code == 400
    assert "Invalid product metadata" in response.json()["detail"]
    assert enqueued == []


def test_bulk_import_is_all_or_nothing_when_a_later_file_is_bad(client, auth_headers, enqueued, png_bytes):
    """A bad file at position 1 must fail the request *before* position 0 is queued (no way to cancel it after)."""
    files = [("files", ("a.png", png_bytes, "image/png")), ("files", ("b.gif", b"GIF89a", "image/gif"))]

    response = _import(client, auth_headers, [_item("bag-a"), _item("bag-b")], files)

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "products[1]" in detail and "'bag-b'" in detail and "unsupported file type" in detail
    assert enqueued == []


def test_bulk_import_rejects_an_empty_file_without_queuing_anything(client, auth_headers, enqueued, png_bytes):
    files = [("files", ("a.png", png_bytes, "image/png")), ("files", ("b.png", b"", "image/png"))]

    response = _import(client, auth_headers, [_item("bag-a"), _item("bag-b")], files)

    assert response.status_code == 400
    assert "empty" in response.json()["detail"]
    assert enqueued == []
