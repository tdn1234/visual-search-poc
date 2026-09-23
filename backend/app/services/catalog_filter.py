"""Catalog filtering by category/color match.

Responsibility: narrow a list of `ProductRecord` down to the ones
matching an optional category and/or color, case-insensitively. Pure,
no I/O.

Two different match rules live here, because the two callers want
different semantics for "both filters given":

- `api/products.py` (plain metadata browsing/faceting) uses AND:
  passing both narrows the results ("Shoes that are also red").
- `api/search.py` (restricting an image search) uses OR: passing both
  widens the candidate pool to "shares the category, or shares the
  color" -- narrowing to AND there would mean color alone, or category
  alone, couldn't be used to broaden an image search at all.
"""

from __future__ import annotations

from app.schemas.search import ProductRecord


def _category_matches(product: ProductRecord, category: str | None) -> bool:
    return category is None or product.category.lower() == category.lower()


def _color_matches(product: ProductRecord, color: str | None) -> bool:
    return color is None or (product.color is not None and product.color.lower() == color.lower())


def filter_by_category_and_color(
    catalog: list[ProductRecord],
    category: str | None = None,
    color: str | None = None,
) -> list[ProductRecord]:
    """Return catalog entries matching category AND color (each if given).

    Case-insensitive, exact match. A product with no `color` set never
    matches a `color` filter. Passing neither filter returns `catalog`
    unchanged; passing both narrows to products satisfying both.

    Args:
        catalog: Products to filter.
        category: If given, only keep products in this category.
        color: If given, only keep products with this color.

    Returns:
        The matching subset of `catalog`, preserving order.
    """
    return [
        product
        for product in catalog
        if _category_matches(product, category) and _color_matches(product, color)
    ]


def filter_by_category_or_color(
    catalog: list[ProductRecord],
    category: str | None = None,
    color: str | None = None,
) -> list[ProductRecord]:
    """Return catalog entries matching category OR color (whichever given).

    Case-insensitive, exact match. A product with no `color` set never
    matches a `color` filter.

    - Only `category` given: keep products in that category.
    - Only `color` given: keep products with that color.
    - Both given: keep products matching *either* -- e.g.
      `category="Shoes", color="red"` keeps every shoe plus every red
      item, not just red shoes.
    - Neither given: `catalog` is returned unchanged.

    Args:
        catalog: Products to filter.
        category: Optional category to match.
        color: Optional color to match.

    Returns:
        The matching subset of `catalog`, preserving order.
    """
    if category is None and color is None:
        return catalog

    def matches(product: ProductRecord) -> bool:
        category_match = category is not None and product.category.lower() == category.lower()
        color_match = color is not None and product.color is not None and product.color.lower() == color.lower()
        return category_match or color_match

    return [product for product in catalog if matches(product)]
