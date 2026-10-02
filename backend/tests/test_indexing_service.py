"""IndexingService: catalog scanning, single-product add, and cleanup on failure (Postgres faked)."""

import json

import pytest
from psycopg import errors as psycopg_errors

from app.services.indexing_service import IndexingService
from tests.fakes import FakeEmbeddingService, FakePool


@pytest.fixture
def catalog(tmp_path):
    """`<tmp>/catalog` -- image_path is stored relative to its parent, like the real project root."""
    path = tmp_path / "catalog"
    path.mkdir()
    return path


def _make_product(catalog, sku, *, metadata="default", image_name="image.png", image=b"img"):
    folder = catalog / sku
    folder.mkdir()
    if image_name:
        (folder / image_name).write_bytes(image)
    if metadata == "default":
        metadata = {"sku": sku, "name": sku.title(), "price": 10.0, "category": "Shoes"}
    if isinstance(metadata, dict):
        (folder / "metadata.json").write_text(json.dumps(metadata))
    elif isinstance(metadata, str):
        (folder / "metadata.json").write_text(metadata)  # e.g. deliberately invalid JSON
    return folder


# --- build_index ------------------------------------------------------------


def test_build_index_embeds_every_valid_product_and_replaces_the_table(catalog):
    _make_product(catalog, "shoe-red", metadata={"sku": "shoe-red", "name": "Red", "price": 5, "category": "Shoes", "color": "red"})
    _make_product(catalog, "bag-x", image_name="image.jpg")
    pool = FakePool()
    embedder = FakeEmbeddingService(embedding=[0.7, 0.3])

    records = IndexingService(embedder, pool).build_index(catalog)

    assert [r.sku for r in records] == ["bag-x", "shoe-red"]  # sorted by folder name
    red = next(r for r in records if r.sku == "shoe-red")
    assert red.color == "red"
    assert red.image_path == "catalog/shoe-red/image.png"
    assert red.embedding == [0.7, 0.3]

    kinds = [(kind, " ".join(sql.split())) for kind, sql, _ in pool.calls]
    assert kinds[0] == ("execute", "TRUNCATE TABLE products")
    assert kinds[1][0] == "executemany"
    assert len(pool.calls[1][2]) == 2  # one INSERT per product


def test_build_index_skips_broken_products_but_keeps_good_ones(catalog):
    _make_product(catalog, "good")
    _make_product(catalog, "no-metadata", metadata=None)
    _make_product(catalog, "no-image", image_name=None)
    _make_product(catalog, "missing-fields", metadata={"sku": "missing-fields", "name": "x"})
    _make_product(catalog, "bad-json", metadata="{not json")
    (catalog / "stray-file.txt").write_text("ignored: not a folder")

    records = IndexingService(FakeEmbeddingService(), FakePool()).build_index(catalog)

    assert [r.sku for r in records] == ["good"]


def test_build_index_raises_if_catalog_dir_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        IndexingService(FakeEmbeddingService(), FakePool()).build_index(tmp_path / "nope")


def test_build_index_raises_if_catalog_has_no_product_folders(catalog):
    with pytest.raises(ValueError, match="No product folders"):
        IndexingService(FakeEmbeddingService(), FakePool()).build_index(catalog)


def test_build_index_raises_and_leaves_table_alone_if_nothing_is_valid(catalog):
    _make_product(catalog, "broken", metadata=None)
    pool = FakePool()

    with pytest.raises(ValueError, match="No valid products"):
        IndexingService(FakeEmbeddingService(), pool).build_index(catalog)

    assert pool.calls == []  # never truncated a table it couldn't refill


def test_build_index_skips_products_whose_image_cannot_be_embedded(catalog):
    _make_product(catalog, "corrupt")
    embedder = FakeEmbeddingService(error=ValueError("File is not a valid image"))

    with pytest.raises(ValueError, match="No valid products"):
        IndexingService(embedder, FakePool()).build_index(catalog)


# --- add_product ------------------------------------------------------------


def _add(service, catalog, **overrides):
    kwargs = dict(
        catalog_dir=catalog,
        sku="shoe-new",
        name="New Shoe",
        price=59.5,
        category="Shoes",
        color="green",
        image_bytes=b"png-bytes",
        image_filename="image.png",
    )
    kwargs.update(overrides)
    return service.add_product(**kwargs)


def test_add_product_writes_catalog_folder_and_db_row(catalog):
    pool = FakePool()
    embedder = FakeEmbeddingService(embedding=[0.1, 0.9])

    record = _add(IndexingService(embedder, pool), catalog)

    folder = catalog / "shoe-new"
    assert (folder / "image.png").read_bytes() == b"png-bytes"
    assert json.loads((folder / "metadata.json").read_text()) == {
        "sku": "shoe-new",
        "name": "New Shoe",
        "price": 59.5,
        "category": "Shoes",
        "color": "green",
    }
    assert embedder.bytes_calls == [b"png-bytes"]
    assert record.image_path == "catalog/shoe-new/image.png"

    _, sql, params = pool.calls[0]
    assert "INSERT INTO products" in sql
    assert params == ("shoe-new", "New Shoe", 59.5, "Shoes", "green", "catalog/shoe-new/image.png", [0.1, 0.9])


@pytest.mark.parametrize("color", [None, ""])
def test_add_product_omits_color_from_metadata_when_not_given(catalog, color):
    _add(IndexingService(FakeEmbeddingService(), FakePool()), catalog, color=color)

    metadata = json.loads((catalog / "shoe-new" / "metadata.json").read_text())
    assert "color" not in metadata


def test_add_product_refuses_existing_sku_and_leaves_the_existing_folder_untouched(catalog):
    existing = _make_product(catalog, "shoe-new", image=b"original")
    pool = FakePool()

    with pytest.raises(FileExistsError):
        _add(IndexingService(FakeEmbeddingService(), pool), catalog)

    assert (existing / "image.png").read_bytes() == b"original"
    assert pool.calls == []


def test_add_product_removes_its_folder_if_the_image_is_invalid(catalog):
    embedder = FakeEmbeddingService(error=ValueError("Uploaded file is not a valid image"))

    with pytest.raises(ValueError):
        _add(IndexingService(embedder, FakePool()), catalog)

    assert not (catalog / "shoe-new").exists()


def test_add_product_maps_a_db_unique_violation_to_file_exists_and_cleans_up(catalog):
    """Two concurrent requests for the same new sku: the loser hits the DB's PK, not the folder check."""
    pool = FakePool(execute_error=psycopg_errors.UniqueViolation("duplicate key"))

    with pytest.raises(FileExistsError, match="already exists"):
        _add(IndexingService(FakeEmbeddingService(), pool), catalog)

    assert not (catalog / "shoe-new").exists()


def test_add_product_cleans_up_and_reraises_unexpected_db_errors(catalog):
    pool = FakePool(execute_error=RuntimeError("connection lost"))

    with pytest.raises(RuntimeError, match="connection lost"):
        _add(IndexingService(FakeEmbeddingService(), pool), catalog)

    assert not (catalog / "shoe-new").exists()


# --- update_product ---------------------------------------------------------


def _update(service, catalog, **overrides):
    kwargs = dict(
        catalog_dir=catalog, sku="shoe-new", name="Renamed", price=70.0, category="Shoes", color=None,
        image_bytes=None, image_filename=None,
    )
    kwargs.update(overrides)
    return service.update_product(**kwargs)


def test_update_product_metadata_only_keeps_photo_and_embedding(catalog):
    _make_product(catalog, "shoe-new", image=b"original")
    pool = FakePool(results=[[("catalog/shoe-new/image.png", [0.3, 0.4])]])
    embedder = FakeEmbeddingService()

    record = _update(IndexingService(embedder, pool), catalog)

    assert embedder.bytes_calls == []
    _, sql, params = pool.calls[0]
    assert "UPDATE products" in sql and "embedding" not in sql.split("WHERE")[0]
    assert params == ("Renamed", 70.0, "Shoes", None, "shoe-new")
    assert record.embedding == [0.3, 0.4]
    assert (catalog / "shoe-new" / "image.png").read_bytes() == b"original"
    assert json.loads((catalog / "shoe-new" / "metadata.json").read_text()) == {
        "sku": "shoe-new", "name": "Renamed", "price": 70.0, "category": "Shoes",
    }


def test_update_product_with_new_image_reembeds_and_replaces_the_old_file(catalog):
    _make_product(catalog, "shoe-new", image_name="image.jpg", image=b"old")
    pool = FakePool(results=[[("catalog/shoe-new/image.png", [0.1, 0.9])]])
    embedder = FakeEmbeddingService(embedding=[0.1, 0.9])

    _update(IndexingService(embedder, pool), catalog, image_bytes=b"new", image_filename="image.png")

    assert embedder.bytes_calls == [b"new"]
    assert pool.calls[0][2] == ("Renamed", 70.0, "Shoes", None, "catalog/shoe-new/image.png", [0.1, 0.9], "shoe-new")
    assert (catalog / "shoe-new" / "image.png").read_bytes() == b"new"
    assert not (catalog / "shoe-new" / "image.jpg").exists()


def test_update_product_unknown_sku_raises_and_writes_nothing(catalog):
    with pytest.raises(FileNotFoundError):
        _update(IndexingService(FakeEmbeddingService(), FakePool()), catalog)

    assert not (catalog / "shoe-new").exists()


def test_update_product_invalid_image_changes_nothing(catalog):
    _make_product(catalog, "shoe-new", image=b"original")
    pool = FakePool()
    embedder = FakeEmbeddingService(error=ValueError("not a valid image"))

    with pytest.raises(ValueError):
        _update(IndexingService(embedder, pool), catalog, image_bytes=b"x", image_filename="image.png")

    assert pool.calls == []
    assert (catalog / "shoe-new" / "image.png").read_bytes() == b"original"
