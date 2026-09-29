"""Product query service.

Responsibility: every read `POST /search` and `GET /products` need,
run as an actual SQL query against the `products` table -- nothing is
loaded into process memory up front. Nearest-neighbor ranking uses
pgvector's `<=>` cosine-distance operator (backed by the HNSW index
`app.db` creates), so Postgres itself does the ranking; this class
only shapes the SQL and maps rows back to Pydantic models.

This is the read-path counterpart to `IndexingService`, which owns the
*write* path (`build_index`, run offline by `scripts/build_embeddings.py`).
Neither loads the whole catalog into memory, unlike the JSON-file
version this replaced.
"""

from __future__ import annotations

from psycopg_pool import ConnectionPool

from app.config import TOP_K_RESULTS
from app.schemas.search import ProductSummary, SearchResult


class ProductQueryService:
    """Read-only SQL queries against the `products` table."""

    def __init__(self, db_pool: ConnectionPool) -> None:
        """Store a reference to an already-open pool (schema already ensured to exist)."""
        self._db_pool = db_pool

    def count(self) -> int:
        """Return how many products are currently indexed."""
        with self._db_pool.connection() as conn:
            row = conn.execute("SELECT count(*) FROM products").fetchone()
        return row[0] if row else 0

    def distinct_categories(self) -> list[str]:
        """Every distinct `category` value currently in the catalog, sorted."""
        with self._db_pool.connection() as conn:
            rows = conn.execute("SELECT DISTINCT category FROM products ORDER BY category").fetchall()
        return [row[0] for row in rows]

    def distinct_colors(self) -> list[str]:
        """Every distinct non-null `color` value currently in the catalog, sorted."""
        with self._db_pool.connection() as conn:
            rows = conn.execute(
                "SELECT DISTINCT color FROM products WHERE color IS NOT NULL ORDER BY color"
            ).fetchall()
        return [row[0] for row in rows]

    def search_similar(
        self,
        query_embedding: list[float],
        top_k: int = TOP_K_RESULTS,
        category: str | None = None,
        color: str | None = None,
    ) -> list[SearchResult]:
        """Nearest-neighbor search via pgvector, optionally restricted by category/color.

        `category`/`color` combine as **OR** -- matching the old
        `catalog_filter.filter_by_category_or_color` semantics this
        replaces: passing both broadens the candidate pool ("shares
        the category, or shares the color") rather than narrowing it,
        since these represent *one* uploaded image's own predicted
        category/color, not two independent facets.

        Args:
            query_embedding: The (already adapter-applied) embedding
                of the uploaded search image.
            top_k: How many results to return.
            category: If given, only consider products in this
                category (case-insensitive), or matching `color`.
            color: If given, only consider products with this color
                (case-insensitive), or matching `category`.

        Returns:
            Up to `top_k` `SearchResult`, ordered by ascending cosine
            distance (i.e. descending similarity). Empty if filtering
            leaves no candidates.
        """
        params: list = [query_embedding]

        where_sql = ""
        if category is not None or color is not None:
            clauses = []
            if category is not None:
                clauses.append("category ILIKE %s")
                params.append(category)
            if color is not None:
                clauses.append("color ILIKE %s")
                params.append(color)
            where_sql = "WHERE " + " OR ".join(clauses)

        params.extend([query_embedding, top_k])

        # Explicit `::vector` cast: pgvector's array->vector cast is only
        # automatic in assignment context (e.g. INSERT into a vector
        # column), not inside an operator expression like `<=>` -- without
        # it Postgres binds the plain Python list as `double precision[]`
        # and `vector <=> double precision[]` has no matching operator.
        sql = f"""
            SELECT sku, name, price, category, embedding <=> %s::vector AS distance
            FROM products
            {where_sql}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        with self._db_pool.connection() as conn:
            rows = conn.execute(sql, params).fetchall()

        return [
            SearchResult(
                sku=row[0],
                name=row[1],
                price=row[2],
                category=row[3],
                # pgvector's `<=>` is cosine *distance* (1 - cosine similarity)
                # for normalized vectors; flip it back to a similarity score.
                score=round(1.0 - float(row[4]), 4),
            )
            for row in rows
        ]

    def list_products(self, category: str | None = None, color: str | None = None) -> list[ProductSummary]:
        """Plain metadata filter, no embeddings/ranking involved.

        `category`/`color` combine as **AND** -- both given narrows to
        that exact facet combination (e.g. "Shoes that are also red"),
        the ordinary meaning of stacking filters in a faceted browse UI.

        Args:
            category: If given, only return products in this category
                (case-insensitive, exact).
            color: If given, only return products with this color
                (case-insensitive, exact). A product with no color set
                never matches.

        Returns:
            Every matching product, sorted by sku.
        """
        clauses = []
        params: list = []
        if category is not None:
            clauses.append("category ILIKE %s")
            params.append(category)
        if color is not None:
            clauses.append("color ILIKE %s")
            params.append(color)
        where_sql = ("WHERE " + " AND ".join(clauses)) if clauses else ""

        sql = f"SELECT sku, name, price, category, color FROM products {where_sql} ORDER BY sku"
        with self._db_pool.connection() as conn:
            rows = conn.execute(sql, params).fetchall()

        return [
            ProductSummary(sku=row[0], name=row[1], price=row[2], category=row[3], color=row[4])
            for row in rows
        ]
