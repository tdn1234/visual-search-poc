"""ProductQueryService: the SQL it builds and how rows map back to models (Postgres faked)."""

from app.services.product_query_service import ProductQueryService
from tests.fakes import FakePool, normalize_sql

EMBEDDING = [0.1, 0.2, 0.3]


def _last_call(pool: FakePool) -> tuple[str, list]:
    _, sql, params = pool.calls[-1]
    return normalize_sql(sql), params


# --- count / distinct -------------------------------------------------------


def test_count_returns_the_row_count():
    assert ProductQueryService(FakePool(results=[[(7,)]])).count() == 7


def test_count_is_zero_when_no_row_comes_back():
    assert ProductQueryService(FakePool(results=[[]])).count() == 0


def test_distinct_categories_and_colors_flatten_rows():
    pool = FakePool(results=[[("Bags",), ("Shoes",)], [("black",), ("red",)]])
    service = ProductQueryService(pool)

    assert service.distinct_categories() == ["Bags", "Shoes"]
    assert service.distinct_colors() == ["black", "red"]
    # Colors query must exclude products with no color.
    assert "WHERE color IS NOT NULL" in normalize_sql(pool.calls[1][1])


# --- search_similar ---------------------------------------------------------


def test_search_similar_without_filters_has_no_where_clause():
    pool = FakePool(results=[[]])
    ProductQueryService(pool).search_similar(EMBEDDING, top_k=5)

    sql, params = _last_call(pool)
    assert "WHERE" not in sql
    assert "ORDER BY embedding <=> %s::vector LIMIT %s" in sql
    assert params == [EMBEDDING, EMBEDDING, 5]


def test_search_similar_category_only():
    pool = FakePool(results=[[]])
    ProductQueryService(pool).search_similar(EMBEDDING, top_k=3, category="Shoes")

    sql, params = _last_call(pool)
    assert "WHERE category ILIKE %s" in sql
    assert "color" not in sql.split("FROM")[1]
    assert params == [EMBEDDING, "Shoes", EMBEDDING, 3]


def test_search_similar_category_and_color_combine_with_or():
    """OR, not AND: both come from ONE uploaded image's predicted labels, so they broaden the pool."""
    pool = FakePool(results=[[]])
    ProductQueryService(pool).search_similar(EMBEDDING, top_k=5, category="Shoes", color="red")

    sql, params = _last_call(pool)
    assert "WHERE category ILIKE %s OR color ILIKE %s" in sql
    assert params == [EMBEDDING, "Shoes", "red", EMBEDDING, 5]


def test_search_similar_converts_cosine_distance_to_similarity_score():
    pool = FakePool(results=[[("a", "A", 10.0, "Shoes", 0.25), ("b", "B", 20.0, "Bags", 0.123456)]])

    results = ProductQueryService(pool).search_similar(EMBEDDING, top_k=2)

    assert [(r.sku, r.name, r.price, r.category) for r in results] == [
        ("a", "A", 10.0, "Shoes"),
        ("b", "B", 20.0, "Bags"),
    ]
    assert [r.score for r in results] == [0.75, 0.8765]  # 1 - distance, rounded to 4 dp


def test_search_similar_returns_empty_list_when_nothing_matches():
    assert ProductQueryService(FakePool(results=[[]])).search_similar(EMBEDDING, top_k=5) == []


def test_user_supplied_filter_values_are_bound_parameters_not_interpolated():
    pool = FakePool(results=[[]])
    hostile = "Shoes'; DROP TABLE products; --"
    ProductQueryService(pool).search_similar(EMBEDDING, top_k=5, category=hostile)

    sql, params = _last_call(pool)
    assert "DROP TABLE" not in sql
    assert hostile in params


# --- list_products ----------------------------------------------------------


def test_list_products_without_filters_returns_everything_sorted_by_sku():
    pool = FakePool(results=[[("a", "A", 1.0, "Bags", None), ("b", "B", 2.0, "Shoes", "red")]])

    results = ProductQueryService(pool).list_products()

    sql, params = _last_call(pool)
    assert "WHERE" not in sql
    assert sql.endswith("ORDER BY sku")
    assert params == []
    assert [(r.sku, r.color) for r in results] == [("a", None), ("b", "red")]


def test_list_products_filters_combine_with_and():
    """AND, unlike search: stacked facet filters ("Shoes that are also red") narrow the result."""
    pool = FakePool(results=[[]])
    ProductQueryService(pool).list_products(category="Shoes", color="red")

    sql, params = _last_call(pool)
    assert "WHERE category ILIKE %s AND color ILIKE %s" in sql
    assert params == ["Shoes", "red"]


def test_list_products_max_price_is_inclusive_and_parameterized():
    pool = FakePool(results=[[]])
    ProductQueryService(pool).list_products(color="red", max_price=80.0)

    sql, params = _last_call(pool)
    assert "WHERE color ILIKE %s AND price <= %s" in sql
    assert params == ["red", 80.0]


def test_list_products_single_filter():
    pool = FakePool(results=[[]])
    ProductQueryService(pool).list_products(color="red")

    sql, params = _last_call(pool)
    assert "WHERE color ILIKE %s" in sql
    assert "category" not in sql.split("FROM")[1]
    assert params == ["red"]


# --- SQL logging ------------------------------------------------------------


def test_sql_and_params_are_debug_logged_before_running_and_embedding_is_summarized(caplog):
    import logging

    pool = FakePool(results=[[]])
    with caplog.at_level(logging.DEBUG, logger="app.services.product_query_service"):
        ProductQueryService(pool).search_similar([0.1] * 512, top_k=5, category="Shoes")

    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("SQL:"))
    assert "WHERE category ILIKE %s" in line and "LIMIT %s" in line
    assert "'Shoes'" in line and "<512 values>" in line
    assert "0.1, 0.1" not in line  # raw vector must not be dumped


def test_get_product_and_list_products_log_their_sql(caplog):
    import logging

    pool = FakePool(results=[[], []])
    service = ProductQueryService(pool)
    with caplog.at_level(logging.DEBUG, logger="app.services.product_query_service"):
        service.get_product("shoe-red")
        service.list_products(color="red", max_price=80)

    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("SQL:")]
    assert "WHERE sku = %s | params=['shoe-red']" in lines[0]
    assert "color ILIKE %s AND price <= %s" in lines[1] and "['red', 80]" in lines[1]
