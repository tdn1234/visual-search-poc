"""Tools the local AI agent can call.

Each tool is a plain function over an existing service plus a JSON
schema the LLM reads to decide when/how to call it. Tools are
read-only and return small, JSON-serializable dicts (never embeddings)
so results fit comfortably in a small model's context window.

Arguments come from the LLM and are untrusted: they are validated with
pydantic before anything touches the database, and a bad call comes
back as an `{"error": ...}` dict the model can read and correct,
rather than an exception that would abort the agent loop.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.config import CATEGORY_MAX_LENGTH, COLOR_MAX_LENGTH, MAX_PRICE, SHOPPER_ID_PATTERN, SKU_PATTERN
from app.services.embedding_service import EmbeddingService
from app.services.product_query_service import ProductQueryService
from app.services.recommendation_service import RecommendationService

logger = logging.getLogger(__name__)

# Most products a single tool call returns to the model. Keeps the tool
# result (and therefore the LLM context) small.
SEARCH_PRODUCTS_MAX_RESULTS: int = 10
SEARCH_SIMILAR_MAX_RESULTS: int = 5
RECOMMENDATIONS_MAX_RESULTS: int = 5

# Uploaded images kept server-side for the agent; oldest are evicted first.
IMAGE_STORE_MAX_ITEMS: int = 50


def _format_validation_error(exc: ValidationError) -> str:
    # Only field + message: no input echo, keeps the error short for the model.
    details = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
    return f"Invalid arguments: {details}"


class ImageStore:
    """Server-side holder for uploaded images, addressed by opaque refs like "img_1".

    Image bytes must never pass through the LLM (huge, useless as text);
    the API layer `add`s an upload, tells the model only the ref, and the
    tool `get`s the bytes back. Bounded and thread-safe (FastAPI runs
    sync handlers in a thread pool).
    """

    def __init__(self, max_items: int = IMAGE_STORE_MAX_ITEMS) -> None:
        self._max_items = max_items
        self._images: OrderedDict[str, bytes] = OrderedDict()
        self._counter = 0
        self._lock = threading.Lock()

    def add(self, image_bytes: bytes) -> str:
        """Store already-validated image bytes and return their ref."""
        with self._lock:
            self._counter += 1
            ref = f"img_{self._counter}"
            self._images[ref] = image_bytes
            while len(self._images) > self._max_items:
                self._images.popitem(last=False)
            return ref

    def get(self, ref: str) -> bytes | None:
        with self._lock:
            return self._images.get(ref)


class SearchProductsArgs(BaseModel):
    """Validated arguments for the `search_products` tool."""

    # Reject arguments the schema doesn't declare -- small models sometimes invent them.
    model_config = ConfigDict(extra="forbid")

    category: str | None = Field(default=None, min_length=1, max_length=CATEGORY_MAX_LENGTH)
    color: str | None = Field(default=None, min_length=1, max_length=COLOR_MAX_LENGTH)
    max_price: float | None = Field(default=None, gt=0, le=MAX_PRICE, allow_inf_nan=False)


SEARCH_PRODUCTS_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_products",
        "description": (
            "Search the product catalog by metadata filters. All filters are optional and "
            "combine with AND. Use this for requests like 'red shoes under 80 dollars'. "
            f"Returns at most {SEARCH_PRODUCTS_MAX_RESULTS} products."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "Product category, e.g. 'Shoes'."},
                "color": {"type": "string", "description": "Product color, e.g. 'red'."},
                "max_price": {"type": "number", "description": "Maximum price (inclusive)."},
            },
        },
    },
}


def search_products(product_query_service: ProductQueryService, arguments: dict[str, Any]) -> dict[str, Any]:
    """Run the `search_products` tool.

    Args:
        product_query_service: Read-only query service to run the filter against.
        arguments: Raw arguments as produced by the LLM (untrusted).

    Returns:
        `{"count": <total matches>, "products": [...]}` with at most
        SEARCH_PRODUCTS_MAX_RESULTS products, or `{"error": "..."}` if the
        arguments were invalid.
    """
    try:
        args = SearchProductsArgs.model_validate(arguments)
    except ValidationError as exc:
        return {"error": _format_validation_error(exc)}

    products = product_query_service.list_products(
        category=args.category, color=args.color, max_price=args.max_price
    )
    logger.debug("search_products(%s) -> %d match(es)", args.model_dump(exclude_none=True), len(products))

    return {
        "count": len(products),
        "products": [
            {"sku": p.sku, "name": p.name, "price": p.price, "category": p.category, "color": p.color}
            for p in products[:SEARCH_PRODUCTS_MAX_RESULTS]
        ],
    }


class SearchSimilarArgs(BaseModel):
    """Validated arguments for the `search_similar_to_image` tool."""

    model_config = ConfigDict(extra="forbid")

    image_ref: str = Field(pattern=r"^img_\d{1,9}$")


SEARCH_SIMILAR_TO_IMAGE_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_similar_to_image",
        "description": (
            "Find catalog products that look visually similar to an image the user uploaded. "
            "Use this when the user message mentions an uploaded image (an image_ref such as 'img_1'). "
            "Results are ranked by visual similarity score (0 to 1, higher is more similar). "
            f"Returns at most {SEARCH_SIMILAR_MAX_RESULTS} products."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "image_ref": {
                    "type": "string",
                    "description": "Reference to the uploaded image, exactly as given in the conversation, e.g. 'img_1'.",
                },
            },
            "required": ["image_ref"],
        },
    },
}


def search_similar_to_image(
    product_query_service: ProductQueryService,
    embedding_service: EmbeddingService,
    image_store: ImageStore,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Run the `search_similar_to_image` tool.

    The model only ever sees the opaque `image_ref`; the bytes are looked
    up in `image_store` here.

    Returns:
        `{"products": [{sku, name, price, category, score}, ...]}` (at most
        SEARCH_SIMILAR_MAX_RESULTS), or `{"error": "..."}`.
    """
    try:
        args = SearchSimilarArgs.model_validate(arguments)
    except ValidationError as exc:
        return {"error": _format_validation_error(exc)}

    image_bytes = image_store.get(args.image_ref)
    if image_bytes is None:
        return {"error": f"Unknown image_ref '{args.image_ref}'. Ask the user to upload the image again."}

    try:
        embedding = embedding_service.embed_image_bytes(image_bytes)
    except ValueError as exc:
        return {"error": f"Could not read the image: {exc}"}

    results = product_query_service.search_similar(embedding, top_k=SEARCH_SIMILAR_MAX_RESULTS)
    return {
        "products": [
            {"sku": r.sku, "name": r.name, "price": r.price, "category": r.category, "score": r.score}
            for r in results
        ]
    }


class GetRecommendationsArgs(BaseModel):
    """Validated arguments for the `get_recommendations` tool."""

    model_config = ConfigDict(extra="forbid")

    shopper_id: str = Field(pattern=SHOPPER_ID_PATTERN)


GET_RECOMMENDATIONS_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_recommendations",
        "description": (
            "Get personalized product recommendations for a shopper based on their browsing and "
            "purchase history. Use this for requests like 'what would I like?'. The result's "
            "'strategy' is 'personalized', or 'popular' when the shopper has no history yet "
            f"(then these are generally popular items). Returns at most {RECOMMENDATIONS_MAX_RESULTS} products."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "shopper_id": {
                    "type": "string",
                    "description": "The current shopper's id, exactly as given in the conversation.",
                },
            },
            "required": ["shopper_id"],
        },
    },
}


def get_recommendations(
    recommendation_service: RecommendationService, arguments: dict[str, Any]
) -> dict[str, Any]:
    """Run the `get_recommendations` tool.

    Returns:
        `{"strategy": ..., "products": [{sku, name, price, category, score}, ...]}`
        (at most RECOMMENDATIONS_MAX_RESULTS), or `{"error": "..."}`.
    """
    try:
        args = GetRecommendationsArgs.model_validate(arguments)
    except ValidationError as exc:
        return {"error": _format_validation_error(exc)}

    strategy, items = recommendation_service.recommend(args.shopper_id, limit=RECOMMENDATIONS_MAX_RESULTS)
    return {
        "strategy": strategy,
        "products": [
            {"sku": i.sku, "name": i.name, "price": i.price, "category": i.category, "score": i.score}
            for i in items
        ],
    }


class GetProductArgs(BaseModel):
    """Validated arguments for the `get_product` tool."""

    model_config = ConfigDict(extra="forbid")

    sku: str = Field(pattern=SKU_PATTERN)


GET_PRODUCT_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_product",
        "description": (
            "Get the details (name, price, category, color) of one product by its SKU. "
            "Use this when the user asks about a specific product or you need details of a SKU "
            "returned by another tool."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sku": {"type": "string", "description": "Product SKU, e.g. 'shoe-red'."},
            },
            "required": ["sku"],
        },
    },
}


def get_product(product_query_service: ProductQueryService, arguments: dict[str, Any]) -> dict[str, Any]:
    """Run the `get_product` tool.

    Returns:
        `{"product": {sku, name, price, category, color}}`, or `{"error": "..."}`
        if the arguments were invalid or the SKU isn't in the catalog.
    """
    try:
        args = GetProductArgs.model_validate(arguments)
    except ValidationError as exc:
        return {"error": _format_validation_error(exc)}

    product = product_query_service.get_product(args.sku)
    if product is None:
        return {"error": f"No product with sku '{args.sku}'."}
    return {"product": product.model_dump()}


# Schemas to hand to the LLM (Ollama/OpenAI "tools" format).
TOOL_SCHEMAS: list[dict[str, Any]] = [
    SEARCH_PRODUCTS_SCHEMA,
    SEARCH_SIMILAR_TO_IMAGE_SCHEMA,
    GET_RECOMMENDATIONS_SCHEMA,
    GET_PRODUCT_SCHEMA,
]
