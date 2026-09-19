"""Indexing service.

Responsibility: build the product index (scan `catalog/`, compute an
embedding per product, write `embeddings.json`) and load that index
back into memory at API startup.

This is the only module that knows about the on-disk catalog layout
(`catalog/<sku>/image.jpg` + `catalog/<sku>/metadata.json`) and about
the embeddings JSON file format.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from app.config import METADATA_FILENAME, SUPPORTED_IMAGE_NAMES
from app.schemas.search import ProductRecord
from app.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class IndexingService:
    """Builds and loads the JSON-backed product embedding index."""

    def __init__(self, embedding_service: EmbeddingService) -> None:
        """Store a reference to the embedding service used to encode images."""
        self._embedding_service = embedding_service

    def build_index(self, catalog_dir: Path, output_file: Path) -> list[ProductRecord]:
        """Scan `catalog_dir`, embed every product image, and persist the index.

        Expected catalog layout::

            catalog/
              shoe-red/
                image.jpg
                metadata.json

        Args:
            catalog_dir: Root directory containing one sub-folder per product.
            output_file: Where to write the resulting embeddings.json.

        Returns:
            The list of `ProductRecord` that was written to disk.

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

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as file:
            json.dump([record.model_dump() for record in records], file, indent=2)

        logger.info("Wrote %d product embeddings to %s", len(records), output_file)
        return records

    def load_index(self, embeddings_file: Path) -> list[ProductRecord]:
        """Load a previously built embeddings.json into memory.

        Args:
            embeddings_file: Path to embeddings.json.

        Returns:
            The list of `ProductRecord` found in the file. Returns an
            empty list (rather than raising) if the file does not
            exist yet, so the API can start up before the catalog has
            been indexed for the first time.
        """
        if not embeddings_file.exists():
            logger.warning(
                "Embeddings file not found at %s. "
                "Run 'python scripts/build_embeddings.py' to create it.",
                embeddings_file,
            )
            return []

        with embeddings_file.open("r", encoding="utf-8") as file:
            raw_records = json.load(file)

        return [ProductRecord(**raw_record) for raw_record in raw_records]

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
            image_path=str(image_path.relative_to(catalog_dir.parent)),
            embedding=embedding,
        )
