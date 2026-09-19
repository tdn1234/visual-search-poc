"""CLI script: build backend/data/embeddings.json from the product catalog.

Usage (run from the `backend/` directory, with the venv active):

    python scripts/build_embeddings.py

What it does:
    1. Scans `catalog/<sku>/` for an image + metadata.json per product.
    2. Loads CLIP once and computes one embedding per product image.
    3. Writes the full result to backend/data/embeddings.json.

This is meant to be re-run any time products are added, removed, or
their photos change. The API only ever *reads* embeddings.json; it
never regenerates it on its own.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

# Allow running this script directly (python scripts/build_embeddings.py)
# by adding backend/ to sys.path, so `import app...` resolves.
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import CATALOG_DIR, EMBEDDINGS_FILE  # noqa: E402
from app.models.clip_model import ClipModel  # noqa: E402
from app.services.embedding_service import EmbeddingService  # noqa: E402
from app.services.indexing_service import IndexingService  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Build the embeddings index and report a summary to stdout."""
    start_time = time.perf_counter()

    logger.info("Catalog directory: %s", CATALOG_DIR)
    logger.info("Output file:       %s", EMBEDDINGS_FILE)

    logger.info("Loading CLIP model (this can take a while on first run, "
                "since it downloads the model from Hugging Face)...")
    clip_model = ClipModel()

    embedding_service = EmbeddingService(clip_model=clip_model)
    indexing_service = IndexingService(embedding_service=embedding_service)

    try:
        records = indexing_service.build_index(
            catalog_dir=CATALOG_DIR,
            output_file=EMBEDDINGS_FILE,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Failed to build index: %s", exc)
        sys.exit(1)

    elapsed = time.perf_counter() - start_time
    logger.info("Done. Indexed %d products in %.1fs.", len(records), elapsed)
    logger.info("Skus indexed: %s", ", ".join(record.sku for record in records))


if __name__ == "__main__":
    main()
