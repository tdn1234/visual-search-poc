"""Request-correlation and access-log middleware.

Responsibility: give every incoming request a short ID, make that ID
available to every log line emitted while handling it (via
`app.logging_config.request_id_var`), and log one line per request
with the outcome and timing -- the "easier to debug" ask this exists
for: `docker-compose logs backend | grep <request-id>` pulls out every
log line for one specific request, across every service it touched
(embedding, classification, the DB query), without threading a
request ID through every function signature by hand.
"""

from __future__ import annotations

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.logging_config import request_id_var

logger = logging.getLogger(__name__)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assign a request ID, log one access-log line, time every request."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Trust an inbound `X-Request-ID` (e.g. from a reverse proxy or
        # a caller doing its own end-to-end tracing) so a single ID
        # threads through this service's logs *and* the caller's,
        # instead of minting an unrelated one at every hop.
        request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])
        token = request_id_var.set(request_id)

        start_time = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            # Log (with the request ID still attached) before resetting --
            # order matters here, not just cleanup.
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.exception(
                "%s %s -> unhandled exception (%.1fms)",
                request.method,
                request.url.path,
                duration_ms,
            )
            request_id_var.reset(token)
            raise

        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "%s %s -> %d (%.1fms)",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        response.headers["X-Request-ID"] = request_id
        request_id_var.reset(token)
        return response
