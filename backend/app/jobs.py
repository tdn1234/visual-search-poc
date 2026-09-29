"""Bulk-import job: the worker's entrypoint for one queued product.

Responsibility: everything the *worker* process needs to turn one
queued (metadata + image bytes) payload into a stored product. Reuses
`IndexingService.add_product` -- the exact same write path
`POST /products` uses synchronously -- so a bulk-imported product is
embedded and stored identically to one added through the
single-product endpoint; nothing about *how* a product gets stored
differs between the two, only *when* (inline vs. queued) and *who*
calls it (the API process vs. a worker process).

`init_services()` must be called once, eagerly, by the worker's own
entrypoint (`scripts/run_worker.py`) *before* it starts consuming from
RabbitMQ -- it loads CLIP once for that worker process's entire
lifetime, exactly like `app.main`'s `lifespan` does for the API
process. Job functions never construct their own services; they reuse
this module-level singleton, since reloading CLIP per job (every few
seconds) would make the queue slower than the synchronous endpoint it
exists to avoid blocking.
"""

from __future__ import annotations

import base64
import json
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

    Must be called exactly once, before the worker starts consuming
    messages (see `scripts/run_worker.py`). Calling `import_product_job`
    before this has run raises `RuntimeError`.
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


def handle_message(body: bytes) -> None:
    """Decode one RabbitMQ message body and import the product it describes.

    AMQP messages are just bytes, so `app.queue.enqueue_import_job`
    JSON-encodes the job (with the image base64-encoded inside it);
    this is the matching decode step. Kept separate from
    `import_product_job` so that function's signature stays plain
    keyword arguments, independent of the wire format.

    Args:
        body: The raw message body, as published by
            `app.queue.enqueue_import_job`.

    Raises:
        Whatever `import_product_job` raises (only genuinely
            unexpected errors -- see its docstring) -- deliberately
            not caught here, so `scripts/run_worker.py` leaves the
            message unacked on failure instead of dropping it.
    """
    payload = json.loads(body)
    import_product_job(
        batch_id=payload["batch_id"],
        item_index=payload["item_index"],
        sku=payload["sku"],
        name=payload["name"],
        price=payload["price"],
        category=payload["category"],
        color=payload["color"],
        image_bytes=base64.b64decode(payload["image_b64"]),
        image_filename=payload["image_filename"],
    )


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
    """Import exactly one product. Runs on a worker process, never the API process.

    Two different failure modes, handled two different ways:

    - **Permanent** failures (duplicate `sku`, an image that isn't
      actually decodable) will fail identically no matter how many
      times they're retried, so they're logged as a `WARNING` and
      swallowed -- there is no status endpoint (see
      docs/architecture.md's "Bulk product import" section), the log
      line, correlated by `batch_id`/`item_index` via `request_id_var`,
      *is* the outcome.
    - **Unexpected** failures (Postgres unreachable, a bug) are logged
      and then *re-raised*, deliberately -- `scripts/run_worker.py`
      only acks a RabbitMQ message after this function returns without
      raising, so a re-raised exception leaves the message unacked and
      RabbitMQ redelivers it once a worker reconnects, instead of
      silently losing that product.

    Args:
        batch_id: The batch this product was queued as part of.
        item_index: This product's position within that batch.
        sku, name, price, category, color: Product metadata.
        image_bytes: Raw bytes of the product photo.
        image_filename: Filename to store the photo under.

    Raises:
        RuntimeError: If `init_services()` was never called on this
            worker process.
        Exception: Whatever unexpected error occurred, after logging it.
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
    except Exception:
        logger.exception(
            "Unexpected error importing product '%s' (batch '%s', item %d) -- message will be redelivered",
            sku, batch_id, item_index,
        )
        raise
    finally:
        request_id_var.reset(token)
