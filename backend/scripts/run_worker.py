"""CLI: run the background worker that consumes queued bulk-import jobs.

Usage (run from the `backend/` directory, with the venv active, and
Postgres + RabbitMQ reachable):

    python scripts/run_worker.py

Loads CLIP once (like the API's own startup) and then blocks, pulling
messages off the `product_import` RabbitMQ queue one at a time -- see
`app/jobs.py` for what each job actually does, and `app/queue.py` for
the producer side (`POST /products/import`). In docker-compose this
runs as its own `worker` service (same image as `backend`, different
command) -- nothing is processed unless that service is running;
publishing a job never fails just because no worker is currently up,
RabbitMQ holds it durably until one is.

Only one message is processed at a time (`prefetch_count=1`) and
deliberately not multithreaded: this worker's CLIP model + Postgres
connection pool are loaded once at startup and held for the process's
whole lifetime, and nothing about either is safe for concurrent use.

If `app.jobs.import_product_job` raises (an *unexpected* error --
known failures like a duplicate sku are caught and logged there, not
raised), that exception is deliberately left to propagate out of this
script entirely rather than being caught here: the in-flight message
is never acked, so RabbitMQ redelivers it once a worker reconnects,
and the process exits non-zero so `docker-compose`'s restart policy
brings up a fresh worker to do that reconnecting. A message is only
ever lost if it's acked, and it's only acked after `handle_message`
returns successfully.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Allow running this script directly (python scripts/run_worker.py)
# by adding backend/ to sys.path, so `import app...` resolves.
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import pika  # noqa: E402
from pika.adapters.blocking_connection import BlockingChannel  # noqa: E402
from pika.spec import Basic, BasicProperties  # noqa: E402

from app import jobs  # noqa: E402
from app.config import PRODUCT_IMPORT_QUEUE_NAME, RABBITMQ_URL  # noqa: E402
from app.logging_config import configure_logging  # noqa: E402

configure_logging()
logger = logging.getLogger(__name__)


def _on_message(
    channel: BlockingChannel,
    method: Basic.Deliver,
    _properties: BasicProperties,
    body: bytes,
) -> None:
    """Process one message, then ack it -- only reached if processing succeeded."""
    jobs.handle_message(body)
    channel.basic_ack(delivery_tag=method.delivery_tag)


def main() -> None:
    """Load CLIP once, then block, processing jobs until killed or a message fails."""
    jobs.init_services()

    connection = pika.BlockingConnection(pika.URLParameters(RABBITMQ_URL))
    channel = connection.channel()
    channel.queue_declare(queue=PRODUCT_IMPORT_QUEUE_NAME, durable=True)
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=PRODUCT_IMPORT_QUEUE_NAME, on_message_callback=_on_message)

    logger.info("Listening on queue '%s'...", PRODUCT_IMPORT_QUEUE_NAME)
    channel.start_consuming()


if __name__ == "__main__":
    main()
