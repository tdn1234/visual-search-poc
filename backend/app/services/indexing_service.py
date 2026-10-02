"""Indexing service.

Responsibility: the *write* path for the product index -- everything
that turns a product photo into a row in the `products` table. Two
distinct flows live here:

- `build_index`: offline, whole-catalog. Scans `catalog/`, embeds
  every product photo, and replaces the entire `products` table. Run
  manually via `scripts/build_embeddings.py`, never by the running API.
- `add_product`: online, single-product. Used by `POST /products` to
  add exactly one new product without touching any other row. It also
  writes into `catalog/<sku>/` (image + metadata.json), not just the
  DB -- otherwise the next offline `build_index` run (which *replaces*
  the whole table from a catalog-folder scan) would silently delete
  any product that only ever existed in the database.

This is the only module that knows about the on-disk catalog layout
(`catalog/<sku>/image.jpg` + `catalog/<sku>/metadata.json`). The read
path (nearest-neighbor search, metadata filtering) is a separate
concern -- see `ProductQueryService`, which queries the same
`products` table (schema owned by `app.db`) directly per request
instead of loading it into memory.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from psycopg import errors as psycopg_errors
from psycopg_pool import ConnectionPool

from app.config import METADATA_FILENAME, SUPPORTED_IMAGE_NAMES
from app.schemas.search import ProductRecord
from app.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class IndexingService:
    """Builds the Postgres/pgvector-backed product embedding index."""

    def __init__(self, embedding_service: EmbeddingService, db_pool: ConnectionPool) -> None:
        """Store references to the embedding service and DB pool used to build the index.

        Args:
            embedding_service: Used to encode catalog images.
            db_pool: An already-open pool from `app.db.create_pool`
                (schema already ensured to exist).
        """
        self._embedding_service = embedding_service
        self._db_pool = db_pool

    def build_index(self, catalog_dir: Path) -> list[ProductRecord]:
        """Scan `catalog_dir`, embed every product image, and replace the Postgres index.

        Expected catalog layout::

            catalog/
              shoe-red/
                image.jpg
                metadata.json

        Args:
            catalog_dir: Root directory containing one sub-folder per product.

        Returns:
            The list of `ProductRecord` that was written to the `products` table.

        Raises:
            FileNotFoundError: If `catalog_dir` does not exist.
            ValueError: If no valid products are found in `catalog_dir`.
        """
        if not catalog_dir.exists():
            raise FileNotFoundError(f"Catalog directory not found: {catalog_dir}")

        product_dirs = sorted(p for p in catalog_dir.iterdir() if p.is_dir())
        if not product_dirs:
            raise ValueError(f"No product folders found inside {catalog_dir}")

        records: list[ProductRecord] = []

        for product_dir in product_dirs:
            try:
                record = self._build_single_record(product_dir, catalog_dir)
                records.append(record)
                logger.info("Indexed product '%s'", record.sku)
            except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError) as exc:
                logger.warning("Skipping '%s': %s", product_dir.name, exc)

        if not records:
            raise ValueError(
                f"No valid products could be indexed from {catalog_dir}. "
                "Each product folder needs an image.jpg and a metadata.json."
            )

        self._replace_all(records)
        logger.info("Wrote %d product embeddings to Postgres", len(records))
        return records

    def add_product(
        self,
        catalog_dir: Path,
        sku: str,
        name: str,
        price: float,
        category: str,
        color: str | None,
        image_bytes: bytes,
        image_filename: str,
    ) -> ProductRecord:
        """Add exactly one new product: persist it to `catalog/<sku>/` and the `products` table.

        Unlike `build_index` (whole-catalog, offline, replaces every
        row), this touches only the one new row -- the online
        counterpart used by `POST /products`. Writing to `catalog_dir`
        too (not just the DB) keeps this product from being silently
        dropped the next time `build_index` rebuilds the whole table
        from a catalog-folder scan.

        On any failure, the `catalog_dir / sku` folder is removed
        again (best-effort) so a partial product never lingers on disk
        without a corresponding DB row.

        Args:
            catalog_dir: Catalog root (e.g. `app.config.CATALOG_DIR`).
            sku: Unique product identifier. Also becomes a literal
                folder name, so callers must have already validated it
                against `app.config.SKU_PATTERN` -- this method trusts
                it rather than re-validating (that's an HTTP-layer
                concern, not a storage one).
            name: Human-readable product name.
            price: Product price.
            category: Product category, e.g. "Shoes".
            color: Dominant product color, or `None`.
            image_bytes: Raw bytes of the product photo.
            image_filename: Filename to store the photo under, e.g.
                `"image.jpg"` -- must be one of `SUPPORTED_IMAGE_NAMES`
                so a later `build_index` scan picks it back up.

        Returns:
            The `ProductRecord` that was written.

        Raises:
            FileExistsError: If `sku` already has a catalog folder, or
                already exists as a DB row (e.g. a race between two
                concurrent requests for the same new sku).
            ValueError: If `image_bytes` is not a valid image.
        """
        product_dir = catalog_dir / sku
        if product_dir.exists():
            raise FileExistsError(f"Product '{sku}' already exists.")

        product_dir.mkdir(parents=True)
        try:
            image_path = product_dir / image_filename
            image_path.write_bytes(image_bytes)

            metadata: dict[str, object] = {
                "sku": sku,
                "name": name,
                "price": price,
                "category": category,
            }
            if color:
                metadata["color"] = color
            with (product_dir / METADATA_FILENAME).open("w", encoding="utf-8") as file:
                json.dump(metadata, file, indent=2)

            embedding = self._embedding_service.embed_image_bytes(image_bytes)
            record = ProductRecord(
                sku=sku,
                name=name,
                price=price,
                category=category,
                color=color,
                image_path=str(image_path.relative_to(catalog_dir.parent)),
                embedding=embedding,
            )
            self._insert_one(record)
        except Exception:
            shutil.rmtree(product_dir, ignore_errors=True)
            raise

        logger.info("Added new product '%s' via POST /products", sku)
        return record

    def update_product(
        self,
        catalog_dir: Path,
        sku: str,
        name: str,
        price: float,
        category: str,
        color: str | None,
        image_bytes: bytes | None = None,
        image_filename: str | None = None,
    ) -> ProductRecord:
        """Replace an existing product's metadata and, optionally, its photo.

        The update is a full replace of the metadata (an omitted `color`
        clears it), matching what a catalog sync sends. Without
        `image_bytes` the stored photo and embedding are kept; with it the
        photo is re-embedded first (so an invalid image fails before
        anything is changed), the DB row is updated, and only then are
        `catalog/<sku>/`'s files rewritten.

        Args:
            catalog_dir: Catalog root (e.g. `app.config.CATALOG_DIR`).
            sku: Existing product identifier (already validated by the caller).
            name, price, category, color: The new metadata.
            image_bytes: New photo bytes, or `None` to keep the current photo.
            image_filename: Filename for the new photo (one of
                `SUPPORTED_IMAGE_NAMES`); required with `image_bytes`.

        Returns:
            The updated `ProductRecord`.

        Raises:
            FileNotFoundError: If `sku` doesn't exist in the DB.
            ValueError: If `image_bytes` is not a valid image.
        """
        product_dir = catalog_dir / sku
        new_embedding = self._embedding_service.embed_image_bytes(image_bytes) if image_bytes is not None else None
        new_image_path = (
            str((product_dir / image_filename).relative_to(catalog_dir.parent))
            if image_bytes is not None and image_filename
            else None
        )

        with self._db_pool.connection() as conn:
            if new_embedding is not None:
                row = conn.execute(
                    """
                    UPDATE products SET name = %s, price = %s, category = %s, color = %s,
                                        image_path = %s, embedding = %s
                    WHERE sku = %s
                    RETURNING image_path, embedding
                    """,
                    (name, price, category, color, new_image_path, new_embedding, sku),
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    UPDATE products SET name = %s, price = %s, category = %s, color = %s
                    WHERE sku = %s
                    RETURNING image_path, embedding
                    """,
                    (name, price, category, color, sku),
                ).fetchone()
        if row is None:
            raise FileNotFoundError(f"Product '{sku}' does not exist.")
        image_path, embedding = row

        # Mirror the change into catalog/<sku>/ so a later full `build_index` keeps it.
        product_dir.mkdir(parents=True, exist_ok=True)
        if image_bytes is not None and image_filename:
            for other in SUPPORTED_IMAGE_NAMES:
                if other != image_filename:
                    (product_dir / other).unlink(missing_ok=True)
            (product_dir / image_filename).write_bytes(image_bytes)
        metadata: dict[str, object] = {"sku": sku, "name": name, "price": price, "category": category}
        if color:
            metadata["color"] = color
        with (product_dir / METADATA_FILENAME).open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)

        logger.info("Updated product '%s'", sku)
        return ProductRecord(
            sku=sku,
            name=name,
            price=price,
            category=category,
            color=color,
            image_path=image_path,
            embedding=list(embedding),
        )

    def _insert_one(self, record: ProductRecord) -> None:
        """Insert exactly one new row. Raises `FileExistsError` if `sku` already exists."""
        with self._db_pool.connection() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO products (sku, name, price, category, color, image_path, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        record.sku,
                        record.name,
                        record.price,
                        record.category,
                        record.color,
                        record.image_path,
                        record.embedding,
                    ),
                )
            except psycopg_errors.UniqueViolation as exc:
                raise FileExistsError(f"Product '{record.sku}' already exists.") from exc

    def _replace_all(self, records: list[ProductRecord]) -> None:
        """Atomically swap the whole `products` table for `records`."""
        with self._db_pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE products")
                cur.executemany(
                    """
                    INSERT INTO products (sku, name, price, category, color, image_path, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    [
                        (
                            record.sku,
                            record.name,
                            record.price,
                            record.category,
                            record.color,
                            record.image_path,
                            record.embedding,
                        )
                        for record in records
                    ],
                )

    def _build_single_record(self, product_dir: Path, catalog_dir: Path) -> ProductRecord:
        """Build one `ProductRecord` from a single `catalog/<sku>/` folder.

        Raises:
            FileNotFoundError: If no supported image or no metadata.json exists.
            ValueError: If metadata.json is missing required fields.
        """
        metadata_path = product_dir / METADATA_FILENAME
        if not metadata_path.exists():
            raise FileNotFoundError(f"missing {METADATA_FILENAME}")

        image_path = next(
            (product_dir / name for name in SUPPORTED_IMAGE_NAMES if (product_dir / name).exists()),
            None,
        )
        if image_path is None:
            raise FileNotFoundError(f"missing image ({', '.join(SUPPORTED_IMAGE_NAMES)})")

        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)

        required_fields = {"sku", "name", "price", "category"}
        missing_fields = required_fields - metadata.keys()
        if missing_fields:
            raise ValueError(f"metadata.json missing fields: {missing_fields}")

        embedding = self._embedding_service.embed_image_file(image_path)

        return ProductRecord(
            sku=metadata["sku"],
            name=metadata["name"],
            price=metadata["price"],
            category=metadata["category"],
            color=metadata.get("color"),
            image_path=str(image_path.relative_to(catalog_dir.parent)),
            embedding=embedding,
        )
