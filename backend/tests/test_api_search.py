"""POST /search: validation, error mapping, and how match_category/match_color drive the query."""

from app.config import TOP_K_RESULTS


def _search(client, headers, png_bytes, params=None, content_type="image/png", content=None):
    body = png_bytes if content is None else content
    return client.post("/search", headers=headers, params=params, files={"file": ("q", body, content_type)})


def test_returns_ranked_results(client, auth_headers, services, png_bytes):
    response = _search(client, auth_headers, png_bytes)

    assert response.status_code == 200
    assert response.json() == {
        "results": [{"sku": "shoe-red", "name": "Red Shoe", "price": 99.0, "category": "Shoes", "score": 0.91}]
    }


def test_embeds_the_upload_and_queries_top_k_without_filters_by_default(client, auth_headers, services, png_bytes):
    services.embedding.embedding = [0.25, 0.75]

    _search(client, auth_headers, png_bytes)

    assert services.embedding.bytes_calls == [png_bytes]
    assert services.product_query.search_calls == [
        {"query_embedding": [0.25, 0.75], "top_k": TOP_K_RESULTS, "category": None, "color": None}
    ]
    assert services.attribute_classifier.calls == []  # no classification unless asked


def test_match_category_classifies_against_catalog_categories_and_filters_by_it(
    client, auth_headers, services, png_bytes
):
    services.attribute_classifier.result = ("Shoes", "red")

    _search(client, auth_headers, png_bytes, params={"match_category": "true"})

    assert services.attribute_classifier.calls == [{"category_labels": ["Bags", "Shoes"], "color_labels": None}]
    query = services.product_query.search_calls[0]
    assert (query["category"], query["color"]) == ("Shoes", "red")


def test_match_color_only_asks_for_color_labels(client, auth_headers, services, png_bytes):
    _search(client, auth_headers, png_bytes, params={"match_color": "true"})

    assert services.attribute_classifier.calls == [{"category_labels": None, "color_labels": ["black", "red"]}]


def test_both_match_flags_request_both_label_sets(client, auth_headers, services, png_bytes):
    _search(client, auth_headers, png_bytes, params={"match_category": "true", "match_color": "true"})

    assert services.attribute_classifier.calls == [
        {"category_labels": ["Bags", "Shoes"], "color_labels": ["black", "red"]}
    ]


def test_webp_is_accepted_for_search(client, auth_headers, services):
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (4, 4)).save(buffer, format="WEBP")

    response = _search(client, auth_headers, None, content_type="image/webp", content=buffer.getvalue())

    assert response.status_code == 200


def test_unsupported_content_type_is_400(client, auth_headers, services):
    response = _search(client, auth_headers, None, content_type="application/pdf", content=b"%PDF")

    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]
    assert services.embedding.bytes_calls == []


def test_empty_upload_is_400(client, auth_headers, services):
    response = _search(client, auth_headers, None, content=b"")

    assert response.status_code == 400
    assert "empty" in response.json()["detail"]


def test_missing_file_field_is_422(client, auth_headers):
    assert client.post("/search", headers=auth_headers).status_code == 422


def test_empty_catalog_is_503_with_a_hint(client, auth_headers, services, png_bytes):
    services.product_query.product_count = 0

    response = _search(client, auth_headers, png_bytes)

    assert response.status_code == 503
    assert "build_embeddings.py" in response.json()["detail"]
    assert services.embedding.bytes_calls == []  # don't burn CLIP inference on a search that can't succeed


def test_undecodable_image_is_400_with_the_embedding_error(client, auth_headers, services, png_bytes):
    services.embedding.error = ValueError("Uploaded file is not a valid image (jpg/png/webp).")

    response = _search(client, auth_headers, png_bytes)

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file is not a valid image (jpg/png/webp)."


def test_unexpected_ranking_failure_is_a_generic_500_that_does_not_leak_details(
    client, auth_headers, services, png_bytes
):
    services.product_query.search_error = RuntimeError("connection to db at 10.0.0.5 refused")

    response = _search(client, auth_headers, png_bytes)

    assert response.status_code == 500
    assert "10.0.0.5" not in response.text
