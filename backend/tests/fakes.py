"""Hand-written test doubles shared across the suite.

Deliberately tiny and dependency-free: the point of these unit tests is
the app's own logic (SQL shape, validation, status codes, error
handling), not Postgres/CLIP/RabbitMQ -- so those are replaced with
fakes that just record what they were asked to do.
"""

from __future__ import annotations

from contextlib import contextmanager

from PIL import Image

from app.schemas.search import ProductSummary, SearchResult


class FakeResult:
    """What `conn.execute(...)` returns: just `fetchone`/`fetchall`."""

    def __init__(self, rows: list[tuple], rowcount: int = 0) -> None:
        self._rows = rows
        self.rowcount = rowcount

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)


class FakeCursor:
    def __init__(self, pool: "FakePool") -> None:
        self._pool = pool

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *exc_info) -> None:
        return None

    def execute(self, sql: str, params=None) -> None:
        self._pool.calls.append(("execute", sql, params))

    def executemany(self, sql: str, seq) -> None:
        self._pool.calls.append(("executemany", sql, list(seq)))


class FakeConnection:
    def __init__(self, pool: "FakePool") -> None:
        self._pool = pool

    def execute(self, sql: str, params=None) -> FakeResult:
        self._pool.calls.append(("execute", sql, params))
        if self._pool.execute_error is not None:
            raise self._pool.execute_error
        return FakeResult(self._pool.results.pop(0) if self._pool.results else [], self._pool.rowcount)

    def cursor(self) -> FakeCursor:
        return FakeCursor(self._pool)


class FakePool:
    """Stands in for `psycopg_pool.ConnectionPool`.

    `results` is a queue: each `conn.execute()` pops the next list of
    rows. `calls` records every statement as `(kind, sql, params)`.
    """

    def __init__(
        self, results: list[list[tuple]] | None = None, execute_error: Exception | None = None, rowcount: int = 0
    ) -> None:
        self.results = list(results or [])
        self.execute_error = execute_error
        self.rowcount = rowcount  # what `conn.execute(...).rowcount` reports (e.g. rows a DELETE removed)
        self.calls: list[tuple] = []

    @contextmanager
    def connection(self):
        yield FakeConnection(self)


def normalize_sql(sql: str) -> str:
    """Collapse whitespace so assertions don't depend on SQL formatting."""
    return " ".join(sql.split())


class FakeClip:
    """Stands in for `ClipModel` -- no torch/transformers model involved."""

    def __init__(self, embedding: list[float] | None = None, classify_result: str = "label") -> None:
        self.embedding = embedding or [1.0, 0.0]
        self.classify_result = classify_result
        self.encode_calls = 0
        self.classify_calls: list[tuple[list[float], dict[str, str]]] = []

    def encode_image(self, image: Image.Image) -> list[float]:
        self.encode_calls += 1
        return list(self.embedding)

    def classify(self, image_embedding: list[float], candidate_texts: dict[str, str]) -> str:
        self.classify_calls.append((image_embedding, candidate_texts))
        return self.classify_result


class FakeEmbeddingService:
    """Stands in for `EmbeddingService` in indexing/API tests."""

    def __init__(self, embedding: list[float] | None = None, error: Exception | None = None) -> None:
        self.embedding = embedding or [0.5, 0.5]
        self.error = error
        self.bytes_calls: list[bytes] = []
        self.file_calls: list = []

    def embed_image_bytes(self, image_bytes: bytes) -> list[float]:
        self.bytes_calls.append(image_bytes)
        if self.error:
            raise self.error
        return list(self.embedding)

    def embed_image_file(self, image_path) -> list[float]:
        self.file_calls.append(image_path)
        if self.error:
            raise self.error
        return list(self.embedding)


class FakeProductQueryService:
    """Stands in for `ProductQueryService` in API tests; records call kwargs."""

    def __init__(self) -> None:
        self.product_count = 3
        self.categories = ["Bags", "Shoes"]
        self.colors = ["black", "red"]
        self.search_results = [SearchResult(sku="shoe-red", name="Red Shoe", price=99.0, category="Shoes", score=0.91)]
        self.list_results = [ProductSummary(sku="shoe-red", name="Red Shoe", price=99.0, category="Shoes", color="red")]
        self.search_error: Exception | None = None
        self.search_calls: list[dict] = []
        self.list_calls: list[dict] = []

    def count(self) -> int:
        return self.product_count

    def distinct_categories(self) -> list[str]:
        return list(self.categories)

    def distinct_colors(self) -> list[str]:
        return list(self.colors)

    def search_similar(self, **kwargs):
        self.search_calls.append(kwargs)
        if self.search_error:
            raise self.search_error
        return self.search_results

    def list_products(self, **kwargs):
        self.list_calls.append(kwargs)
        return self.list_results


class FakeAttributeClassifier:
    """Stands in for `AttributeClassifierService` in API tests."""

    def __init__(self, category: str | None = "Shoes", color: str | None = "red") -> None:
        self.result = (category, color)
        self.calls: list[dict] = []

    def classify(self, image, category_labels=None, color_labels=None):
        self.calls.append({"category_labels": category_labels, "color_labels": color_labels})
        return self.result


class FakeIndexingService:
    """Stands in for `IndexingService` in API tests; records `add_product` kwargs."""

    def __init__(self) -> None:
        self.error: Exception | None = None
        self.calls: list[dict] = []
        self.update_calls: list[dict] = []

    def update_product(self, **kwargs):
        self.update_calls.append(kwargs)
        if self.error:
            raise self.error
        from app.schemas.search import ProductRecord

        return ProductRecord(
            sku=kwargs["sku"],
            name=kwargs["name"],
            price=kwargs["price"],
            category=kwargs["category"],
            color=kwargs["color"],
            image_path=f"catalog/{kwargs['sku']}/image.png",
            embedding=[0.1, 0.2],
        )

    def add_product(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        from app.schemas.search import ProductRecord

        return ProductRecord(
            sku=kwargs["sku"],
            name=kwargs["name"],
            price=kwargs["price"],
            category=kwargs["category"],
            color=kwargs["color"],
            image_path=f"catalog/{kwargs['sku']}/{kwargs['image_filename']}",
            embedding=[0.1, 0.2],
        )


class FakeRecommendationService:
    """Stands in for `RecommendationService` in API tests; records call kwargs."""

    def __init__(self) -> None:
        from app.schemas.recommendation import RecommendationItem

        self.strategy = "personalized"
        self.recommendations = [
            RecommendationItem(sku="shoe-blue", name="Blue Shoe", price=89.0, category="Shoes", score=0.83)
        ]
        self.deleted = 4
        self.error: Exception | None = None
        self.recorded_batches: list[list] = []
        self.recommend_calls: list[dict] = []
        self.delete_calls: list[str] = []

    def record_events(self, events):
        if self.error:
            raise self.error
        self.recorded_batches.append(list(events))
        return len(events)

    def recommend(self, **kwargs):
        if self.error:
            raise self.error
        self.recommend_calls.append(kwargs)
        return self.strategy, self.recommendations

    def delete_shopper_events(self, shopper_id):
        if self.error:
            raise self.error
        self.delete_calls.append(shopper_id)
        return self.deleted


class FakeAgentService:
    """Stands in for `AgentService` in API tests; records what the endpoint passed in."""

    def __init__(self) -> None:
        from app.services.agent_service import AgentResult

        self.result = AgentResult(
            reply="Here you go.",
            steps=[{"tool": "get_product", "arguments": {"sku": "shoe-red"}, "result": {}}],
            products=[{"sku": "shoe-red", "name": "Red Shoe", "price": 99.0, "category": "Shoes", "score": 0.9}],
        )
        self.error: Exception | None = None
        self.chat_calls: list[dict] = []
        self.images: list[bytes] = []

    def add_image(self, image_bytes: bytes) -> str:
        self.images.append(image_bytes)
        return f"img_{len(self.images)}"

    async def chat(self, message, history=None, shopper_id=None):
        if self.error:
            raise self.error
        self.chat_calls.append({"message": message, "history": history, "shopper_id": shopper_id})
        return self.result


class FakeSessionStore:
    def __init__(self) -> None:
        self.sessions: dict[str, list[dict]] = {}
        self.error: Exception | None = None

    def load(self, session_id):
        if self.error:
            raise self.error
        return list(self.sessions.get(session_id, []))

    def append_turn(self, session_id, history, user_message, reply):
        if self.error:
            raise self.error
        self.sessions[session_id] = [
            *history,
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": reply},
        ]
