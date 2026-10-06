"""Tests for the agent's `search_products` tool (no LLM, no DB)."""

from __future__ import annotations

from app.schemas.search import ProductSummary
from app.services.agent_tools import SEARCH_PRODUCTS_MAX_RESULTS, SEARCH_PRODUCTS_SCHEMA, search_products


class FakeQueryService:
    def __init__(self, products: list[ProductSummary]) -> None:
        self._products = products
        self.calls: list[dict] = []

    def list_products(self, category=None, color=None, max_price=None):
        self.calls.append({"category": category, "color": color, "max_price": max_price})
        return self._products


def _product(i: int) -> ProductSummary:
    return ProductSummary(sku=f"sku-{i}", name=f"Shoe {i}", price=10.0 + i, category="Shoes", color="red")


def test_passes_validated_filters_to_service_and_shapes_result():
    service = FakeQueryService([_product(1)])

    result = search_products(service, {"category": "Shoes", "color": "red", "max_price": 80})

    assert service.calls == [{"category": "Shoes", "color": "red", "max_price": 80.0}]
    assert result == {
        "count": 1,
        "products": [{"sku": "sku-1", "name": "Shoe 1", "price": 11.0, "category": "Shoes", "color": "red"}],
    }


def test_no_arguments_means_no_filters():
    service = FakeQueryService([])
    assert search_products(service, {}) == {"count": 0, "products": []}
    assert service.calls == [{"category": None, "color": None, "max_price": None}]


def test_results_are_capped_but_count_is_the_true_total():
    service = FakeQueryService([_product(i) for i in range(SEARCH_PRODUCTS_MAX_RESULTS + 5)])

    result = search_products(service, {})

    assert result["count"] == SEARCH_PRODUCTS_MAX_RESULTS + 5
    assert len(result["products"]) == SEARCH_PRODUCTS_MAX_RESULTS


def test_invalid_arguments_return_error_and_never_query():
    for bad in ({"max_price": -5}, {"max_price": "cheap"}, {"color": ""}, {"unknown": 1}):
        service = FakeQueryService([])
        result = search_products(service, bad)
        assert "error" in result
        assert service.calls == []


def test_schema_declares_only_the_validated_parameters():
    props = SEARCH_PRODUCTS_SCHEMA["function"]["parameters"]["properties"]
    assert set(props) == {"category", "color", "max_price"}


# --- search_similar_to_image / get_recommendations / get_product -------------

from app.schemas.recommendation import RecommendationItem  # noqa: E402
from app.schemas.search import SearchResult  # noqa: E402
from app.services.agent_tools import (  # noqa: E402
    RECOMMENDATIONS_MAX_RESULTS,
    SEARCH_SIMILAR_MAX_RESULTS,
    TOOL_SCHEMAS,
    ImageStore,
    get_product,
    get_recommendations,
    search_similar_to_image,
)


class FakeEmbedding:
    def __init__(self, fail=False):
        self.fail, self.seen = fail, []

    def embed_image_bytes(self, data):
        self.seen.append(data)
        if self.fail:
            raise ValueError("bad image")
        return [0.1, 0.2]


class FakeSimilarService:
    def __init__(self):
        self.calls = []

    def search_similar(self, embedding, top_k):
        self.calls.append((embedding, top_k))
        return [SearchResult(sku="a", name="A", price=5.0, category="Shoes", score=0.9)]

    def get_product(self, sku):
        return _product(1) if sku == "sku-1" else None


def test_image_store_returns_opaque_refs_and_evicts_oldest():
    store = ImageStore(max_items=2)
    r1, r2, r3 = store.add(b"1"), store.add(b"2"), store.add(b"3")
    assert (r1, r2, r3) == ("img_1", "img_2", "img_3")
    assert store.get(r1) is None and store.get(r3) == b"3"


def test_search_similar_looks_up_bytes_by_ref():
    store, emb, svc = ImageStore(), FakeEmbedding(), FakeSimilarService()
    ref = store.add(b"jpegbytes")

    result = search_similar_to_image(svc, emb, store, {"image_ref": ref})

    assert emb.seen == [b"jpegbytes"]
    assert svc.calls == [([0.1, 0.2], SEARCH_SIMILAR_MAX_RESULTS)]
    assert result == {"products": [{"sku": "a", "name": "A", "price": 5.0, "category": "Shoes", "score": 0.9}]}


def test_search_similar_errors_are_returned_not_raised():
    store, svc = ImageStore(), FakeSimilarService()
    ref = store.add(b"x")
    assert "error" in search_similar_to_image(svc, FakeEmbedding(), store, {"image_ref": "img_99"})
    assert "error" in search_similar_to_image(svc, FakeEmbedding(), store, {"image_ref": "not a ref"})
    assert "error" in search_similar_to_image(svc, FakeEmbedding(), store, {})
    assert "error" in search_similar_to_image(svc, FakeEmbedding(fail=True), store, {"image_ref": ref})
    assert svc.calls == []


class FakeRecs:
    def __init__(self):
        self.calls = []

    def recommend(self, shopper_id, limit):
        self.calls.append((shopper_id, limit))
        return "popular", [RecommendationItem(sku="a", name="A", price=5.0, category="Shoes", score=1.0)]


def test_get_recommendations_shapes_result_and_validates():
    svc = FakeRecs()
    result = get_recommendations(svc, {"shopper_id": "user_1"})
    assert svc.calls == [("user_1", RECOMMENDATIONS_MAX_RESULTS)]
    assert result["strategy"] == "popular" and result["products"][0]["sku"] == "a"
    assert "error" in get_recommendations(svc, {"shopper_id": "bad id!"})
    assert len(svc.calls) == 1


def test_get_product_found_missing_and_invalid():
    svc = FakeSimilarService()
    assert get_product(svc, {"sku": "sku-1"})["product"]["name"] == "Shoe 1"
    assert "error" in get_product(svc, {"sku": "nope"})
    assert "error" in get_product(svc, {"sku": "Bad SKU"})


def test_every_tool_schema_has_described_parameters():
    assert [t["function"]["name"] for t in TOOL_SCHEMAS] == [
        "search_products", "search_similar_to_image", "get_recommendations", "get_product",
    ]
    for t in TOOL_SCHEMAS:
        fn = t["function"]
        assert fn["description"]
        for prop in fn["parameters"]["properties"].values():
            assert prop["description"]
