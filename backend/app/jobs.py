"""Bulk-import job: the RQ worker's entrypoint for one queued product.

Responsibility: everything the *worker* process needs to turn one
queued (metadata + image bytes) payload into a stored product. Reuses
`IndexingService.add_product` -- the exact same write path
`POST /products` uses synchronously -- so a bulk-imported product is
embedded and stored identically to one added through the
single-product endpoint; nothing about *how* a product gets stored
differs between the two, only *when* (inline vs. queued) and *who*
calls it (the API process vs. a worker process).

`init_services()` must be called once, eagerly, by the worker's own
entrypoint (`scripts/run_worker.py`) *before* the RQ work loop starts
-- it loads CLIP once for that worker process's entire lifetime,
exactly like `app.main`'s `lifespan` does for the API process. Job
functions never construct their own services; they reuse this
module-level singleton, since reloading CLIP per job (every few
seconds) would make the queue slower than the synchronous endpoint it
exists to avoid blocking.
"""

from __future__ import annotations

import logging

from app.config import CATALOG_DIR, COLOR_ADAPTER_FILE
from app.db import create_pool
from app.logging_config import request_id_var
from app.models.clip_model import ClipModel
from app.models.color_adapter import load_color_adapter
from app.services.embedding_service import EmbeddingService
from app.services.indexing_service import IndexingService

logger = logging.getLogger(__name__)

_indexing_service: IndexingService | None = None


def init_services() -> None:
    """Load CLIP and open the DB pool once, for this worker process's lifetime.

    Must be called exactly once, before the RQ work loop starts (see
    `scripts/run_worker.py`). Calling `import_product_job` before this
    has run raises `RuntimeError`.
    """
    global _indexing_service

    logger.info("Worker: connecting to Postgres...")
    db_pool = create_pool()

    logger.info("Worker: loading CLIP model...")
    clip_model = ClipModel()
    color_adapter = load_color_adapter(COLOR_ADAPTER_FILE)
    embedding_service = EmbeddingService(clip_model=clip_model, color_adapter=color_adapter)

    _indexing_service = IndexingService(embedding_service=embedding_service, db_pool=db_pool)
    logger.info("Worker ready.")


def import_product_job(
    batch_id: str,
    item_index: int,
    sku: str,
    name: str,
    price: float,
    category: str,
    color: str | None,
    image_bytes: bytes,
    image_filename: str,
) -> None:
    """Import exactly one product. Runs on an RQ worker, never the API process.

    Known failures (duplicate sku, corrupt image) are logged and
    swallowed rather than re-raised, so RQ doesn't retry a job that
    would just fail identically every time -- bulk import is
    fire-and-forget by design here (see
    docs/architecture.md's "Bulk product import" section): the log
    line, correlated by `batch_id`/`item_index` via `request_id_var`,
    *is* the outcome. There is no status endpoint to check instead.

    Args:
        batch_id: The batch this product was queued as part of.
        item_index: This product's position within that batch.
        sku, name, price, category, color: Product metadata.
        image_bytes: Raw bytes of the product photo.
        image_filename: Filename to store the photo under.

    Raises:
        RuntimeError: If `init_services()` was never called on this
            worker process.
    """
    if _indexing_service is None:
        raise RuntimeError("app.jobs.init_services() was never called -- see scripts/run_worker.py")

    token = request_id_var.set(f"import-{batch_id}-{item_index}")
    try:
        _indexing_service.add_product(
            catalog_dir=CATALOG_DIR,
            sku=sku,
            name=name,
            price=price,
            category=category,
            color=color,
            image_bytes=image_bytes,
            image_filename=image_filename,
        )
        logger.info("Imported product '%s' (batch '%s', item %d)", sku, batch_id, item_index)
    except FileExistsError as exc:
        logger.warning("Skipped product '%s' (batch '%s', item %d): %s", sku, batch_id, item_index, exc)
    except ValueError as exc:
        logger.warning(
            "Skipped product '%s' (batch '%s', item %d): invalid image: %s", sku, batch_id, item_index, exc
        )
    except Exception:  # noqa: BLE001 - last-resort safety net for the worker loop
        logger.exception("Unexpected error importing product '%s' (batch '%s', item %d)", sku, batch_id, item_index)
    finally:
        request_id_var.reset(token)
