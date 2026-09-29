"""Redis-backed job queue for bulk product imports.

Responsibility: own the single RQ `Queue` instance
`POST /products/import` enqueues into, and the one function
(`enqueue_import_job`) that puts a job on it. This is the *producer*
side only -- imported by the API process (`api/products.py`). The
*consumer* side (`app/jobs.py` + `scripts/run_worker.py`) deliberately
doesn't import this module, and vice versa: the API process never
needs to load CLIP, and the worker process never needs `slowapi`/auth.

Jobs are enqueued by string reference (`"app.jobs.import_product_job"`)
rather than a direct function import, so this module -- and the API
process that imports it -- never needs to import `app.jobs` (which
would pull in `app.services.indexing_service`, `ClipModel`, etc. for
no reason on the API side; the worker resolves that string itself).
"""

from __future__ import annotations

import redis
from rq import Queue

from app.config import PRODUCT_IMPORT_QUEUE_NAME, REDIS_URL

_redis_connection = redis.from_url(REDIS_URL)
import_queue = Queue(PRODUCT_IMPORT_QUEUE_NAME, connection=_redis_connection)


def enqueue_import_job(
    *,
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
    """Queue exactly one product for background import.

    Fire-and-forget: this only confirms Redis accepted the job, not
    that any worker is currently running to pick it up, or that the
    import will ultimately succeed (duplicate skus, corrupt images,
    etc. are only discovered once a worker actually processes the
    job -- see `app.jobs.import_product_job`).

    Args:
        batch_id: Groups every job from one `POST /products/import`
            call, for log correlation.
        item_index: This product's position within its batch.
        sku, name, price, category, color: Product metadata, already
            validated by the API layer (`BulkProductItem`).
        image_bytes: Raw bytes of the product photo.
        image_filename: Filename to store the photo under, e.g.
            `"image.jpg"`.
    """
    import_queue.enqueue(
        "app.jobs.import_product_job",
        batch_id=batch_id,
        item_index=item_index,
        sku=sku,
        name=name,
        price=price,
        category=category,
        color=color,
        image_bytes=image_bytes,
        image_filename=image_filename,
        job_timeout="5m",
    )
