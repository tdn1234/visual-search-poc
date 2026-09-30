"""Request/response schemas: mainly the SKU rule, which doubles as a path-traversal guard."""

import pytest
from pydantic import ValidationError

from app.schemas.search import BulkProductItem, ProductRecord


def _item(**overrides):
    data = {"sku": "shoe-red", "name": "Red Shoe", "price": 99.0, "category": "Shoes"}
    data.update(overrides)
    return BulkProductItem(**data)


@pytest.mark.parametrize("sku", ["shoe-red", "a", "x1", "bag-black-42", "007"])
def test_valid_skus_are_accepted(sku):
    assert _item(sku=sku).sku == sku


@pytest.mark.parametrize(
    "sku",
    [
        "Shoe-Red",     # uppercase
        "-shoe",        # leading hyphen
        "shoe-",        # trailing hyphen
        "shoe--red",    # doubled hyphen
        "shoe red",     # space
        "shoe_red",     # underscore
        "../etc",       # path traversal
        "a/b",          # path separator
        "a\\b",         # windows separator
        "..",           # parent dir
        "",             # empty
    ],
)
def test_unsafe_or_malformed_skus_are_rejected(sku):
    with pytest.raises(ValidationError):
        _item(sku=sku)


@pytest.mark.parametrize("price", [0, -1, -0.01])
def test_price_must_be_positive(price):
    with pytest.raises(ValidationError):
        _item(price=price)


def test_color_is_optional_and_defaults_to_none():
    assert _item().color is None
    assert _item(color="red").color == "red"


@pytest.mark.parametrize("missing", ["sku", "name", "price", "category"])
def test_required_fields(missing):
    data = {"sku": "shoe-red", "name": "Red Shoe", "price": 99.0, "category": "Shoes"}
    del data[missing]
    with pytest.raises(ValidationError):
        BulkProductItem(**data)


def test_product_record_keeps_embedding_and_optional_color():
    record = ProductRecord(
        sku="a", name="A", price=1.0, category="C", image_path="catalog/a/image.png", embedding=[0.1, 0.2]
    )
    assert record.color is None
    assert record.embedding == [0.1, 0.2]
