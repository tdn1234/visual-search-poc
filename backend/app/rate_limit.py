"""Redis-backed request rate limiting.

Responsibility: own the single `slowapi` `Limiter` instance, backed by
Redis (rather than in-process memory) so limits are shared across
every backend replica instead of being tracked separately per worker.
`app/main.py` wires this into the FastAPI app (exception handler +
middleware, once, at startup); `api/search.py`/`api/products.py`
import `limiter` to decorate their own routes with per-endpoint
limits, since different endpoints warrant different limits (CLIP
inference is far more expensive than a plain metadata query).
"""

from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import REDIS_URL

# Keyed by client IP (`get_remote_address`), not API key: this POC has
# exactly one shared API key, so keying by it would rate-limit every
# client as a single bucket instead of limiting each caller individually.
limiter = Limiter(key_func=get_remote_address, storage_uri=REDIS_URL)
