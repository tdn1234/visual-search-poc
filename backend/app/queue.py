"""RabbitMQ-backed job queue for bulk product imports.

Responsibility: own the connection logic the API process uses to
publish bulk-import jobs, and the one function (`enqueue_import_job`)
that does it. This is the *producer* side only -- imported by
`api/products.py`. The *consumer* side (`scripts/run_worker.py` +
`app/jobs.py`) connects to RabbitMQ independently; the two processes
share only the queue's name and RabbitMQ's own message durability, not
any Python object or connection.

A dedicated broker (RabbitMQ), not the Redis already used for rate
limiting, deliberately: rate-limit counters are ephemeral, TTL-based,
and fine to lose on a restart, while a queued-but-not-yet-imported
product is real work a caller (e.g. a Magento export) expects to
actually happen. Coupling that to the same Redis instance -- and the
same failure domain -- as rate limiting would muddy what "Redis is
down" means for this service, and Redis's own semantics (no
per-message ack/redelivery without extra work on top) are a worse fit
for "this must not be silently dropped" than a broker built around
exactly that guarantee. See docs/architecture.md's "Bulk product
import" section for the full reasoning.
"""

from __future__ import annotations

import base64
import json

import pika

from app.config import PRODUCT_IMPORT_QUEUE_NAME, RABBITMQ_URL


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

    Opens a short-lived connection per call rather than holding one
    open across requests -- simpler and safe regardless of how many
    FastAPI worker processes/threads are calling this concurrently,
    and bulk imports are infrequent enough (rate-limited via
    `BULK_IMPORT_RATE_LIMIT`) that per-call connection setup is a
    non-issue.

    Fire-and-forget in the sense that this only confirms RabbitMQ
    *accepted* the message, not that any worker is currently running
    to consume it, or that the import will ultimately succeed
    (duplicate skus, corrupt images, etc. are only discoverable once a
    worker actually processes the job -- see
    `app.jobs.import_product_job`). But *not* fire-and-forget about
    durability: the message is published to a durable queue with
    `delivery_mode=2` (persistent), so it survives a broker restart,
    and a worker crash mid-processing leaves it unacked for RabbitMQ
    to redeliver rather than silently dropping it (see
    `scripts/run_worker.py`).

    Args:
        batch_id: Groups every job from one `POST /products/import`
            call, for log correlation.
        item_index: This product's position within its batch.
        sku, name, price, category, color: Product metadata, already
            validated by the API layer (`BulkProductItem`).
        image_bytes: Raw bytes of the product photo. AMQP messages are
            just bytes, so this is base64-encoded into the JSON body.
        image_filename: Filename to store the photo under, e.g.
            `"image.jpg"`.
    """
    payload = {
        "batch_id": batch_id,
        "item_index": item_index,
        "sku": sku,
        "name": name,
        "price": price,
        "category": category,
        "color": color,
        "image_filename": image_filename,
        "image_b64": base64.b64encode(image_bytes).decode("ascii"),
    }

    connection = pika.BlockingConnection(pika.URLParameters(RABBITMQ_URL))
    try:
        channel = connection.channel()
        channel.queue_declare(queue=PRODUCT_IMPORT_QUEUE_NAME, durable=True)
        channel.basic_publish(
            exchange="",
            routing_key=PRODUCT_IMPORT_QUEUE_NAME,
            body=json.dumps(payload).encode("utf-8"),
            properties=pika.BasicProperties(
                delivery_mode=2,  # persistent -- survives a broker restart
                content_type="application/json",
            ),
        )
    finally:
        connection.close()
