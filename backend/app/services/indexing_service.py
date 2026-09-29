"""Indexing service.

Responsibility: the offline *write* path for the product index --
scan `catalog/`, compute an embedding per product, and replace the
`products` table in Postgres. Run manually via
`scripts/build_embeddings.py`, never by the running API.

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
from pathlib import Path

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
