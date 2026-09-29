"""CLI: run the background worker that consumes queued bulk-import jobs.

Usage (run from the `backend/` directory, with the venv active, and
Postgres + Redis reachable):

    python scripts/run_worker.py

Loads CLIP once (like the API's own startup) and then blocks, pulling
jobs off the `product_import` Redis queue one at a time -- see
`app/jobs.py` for what each job actually does, and `app/queue.py` for
the producer side (`POST /products/import`). In docker-compose this
runs as its own `worker` service (same image as `backend`, different
command) -- nothing is processed unless that service is running;
queuing a job never fails just because no worker is currently up.

Uses RQ's `SimpleWorker`, not the default `Worker`, deliberately: the
default forks a child process per job, and this worker's CLIP model +
Postgres connection pool are loaded once at startup and held for the
process's whole lifetime -- forking after that would duplicate (and
likely corrupt) the live DB connection in the child. `SimpleWorker`
runs jobs in-process, one at a time, no fork, which is exactly the
"load once, reuse for every job" model `app.jobs.init_services()` is
built around.
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

import redis  # noqa: E402
from rq import SimpleWorker  # noqa: E402

from app import jobs  # noqa: E402
from app.config import PRODUCT_IMPORT_QUEUE_NAME, REDIS_URL  # noqa: E402
from app.logging_config import configure_logging  # noqa: E402

configure_logging()
logger = logging.getLogger(__name__)


def main() -> None:
    """Load CLIP once, then block, processing jobs until killed."""
    jobs.init_services()

    connection = redis.from_url(REDIS_URL)
    worker = SimpleWorker([PRODUCT_IMPORT_QUEUE_NAME], connection=connection)

    logger.info("Listening on queue '%s'...", PRODUCT_IMPORT_QUEUE_NAME)
    worker.work()


if __name__ == "__main__":
    main()
