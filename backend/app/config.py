"""Centralized configuration for the Visual Product Search POC.

All paths and tunable constants live here so the rest of the codebase
never hardcodes filesystem paths or magic numbers.
"""

import os
from pathlib import Path

# --- Project layout -------------------------------------------------------
# backend/app/config.py -> parents[2] == visual-search-poc/
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

CATALOG_DIR: Path = PROJECT_ROOT / "catalog"
DATA_DIR: Path = PROJECT_ROOT / "backend" / "data"
COLOR_ADAPTER_FILE: Path = DATA_DIR / "color_adapter.pt"
CATEGORY_CLASSIFIER_FILE: Path = DATA_DIR / "category_classifier.pt"

# --- Model configuration ---------------------------------------------------
CLIP_MODEL_NAME: str = "openai/clip-vit-base-patch32"
EMBEDDING_DIM: int = 512  # clip-vit-base-patch32's image/text feature dimension.

# Force CPU for a laptop-friendly POC. Set to "cuda" manually if you have a GPU.
DEVICE: str = "cpu"

# --- Logging -----------------------------------------------------------------
# Standard `logging` level name ("DEBUG", "INFO", "WARNING", ...). DEBUG
# additionally turns on per-step timing logs (embedding, classification,
# the pgvector query) in the services that do the real work -- see
# app.logging_config.configure_logging and docs/architecture.md's
# "Logging and request correlation" section.
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")

# --- Vector database (pgvector) --------------------------------------------
# The product index (embeddings + metadata) lives in Postgres, not on
# disk. In docker-compose this is overridden to point at the `db`
# service; the default below matches that service's host port mapping
# (5433:5432) so the API also runs against it from a bare host checkout.
DATABASE_URL: str = os.environ.get(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5433/visual_search"
)

# --- Auth & rate limiting ---------------------------------------------------
# Single shared-secret API key (no user accounts in this POC) -- sent by
# clients as the `X-API-Key` header, checked by `app.auth.require_api_key`.
# The default below only works for local/dev use; always override it via
# the `API_KEY` env var (docker-compose does) before exposing this
# anywhere other than your own machine.
API_KEY: str = os.environ.get("API_KEY", "dev-api-key-change-me")

# Redis backs the request-rate limiter (`slowapi`) so limits are shared
# across all backend replicas, not just tracked per-process. Default
# matches docker-compose's `redis` service host port mapping (6380:6379).
# Rate-limit counters only -- the bulk-import queue uses its own,
# separate RabbitMQ broker (RABBITMQ_URL below), not this Redis.
REDIS_URL: str = os.environ.get("REDIS_URL", "redis://localhost:6380/0")

# Per-client-IP limits, in `limits`-library syntax (e.g. "20/minute",
# "100/hour", "5/second"). Applied via `@limiter.limit(...)` on each
# route in api/search.py / api/products.py. Different endpoints get
# different budgets: /search runs CLIP inference (expensive), /products
# is a plain SQL filter (cheap) -- see docs/architecture.md's "Auth and
# rate limiting" section for the reasoning.
SEARCH_RATE_LIMIT: str = os.environ.get("SEARCH_RATE_LIMIT", "20/minute")
PRODUCTS_RATE_LIMIT: str = os.environ.get("PRODUCTS_RATE_LIMIT", "60/minute")
# POST /products (creating a new product) does an embed + a filesystem
# write, not just a SQL query -- closer in cost to /search than to
# GET /products, so it gets its own (tighter) budget.
CREATE_PRODUCT_RATE_LIMIT: str = os.environ.get("CREATE_PRODUCT_RATE_LIMIT", "10/minute")
# POST /products/import reads every uploaded file into memory
# synchronously (bounded by MAX_BULK_IMPORT_ITEMS below) before handing
# off to the queue -- tighter still, since one call can carry many
# products' worth of work.
BULK_IMPORT_RATE_LIMIT: str = os.environ.get("BULK_IMPORT_RATE_LIMIT", "5/minute")

# --- Bulk product import (queue) --------------------------------------------
# POST /products/import enqueues one background job per product onto a
# durable RabbitMQ queue (app/queue.py) instead of embedding
# synchronously in the request -- see docs/architecture.md's "Bulk
# product import" section for why, and specifically why RabbitMQ
# rather than the Redis already used for rate limiting above (a
# deliberately separate broker/failure-domain for durable work vs.
# ephemeral counters). scripts/run_worker.py is the consumer; it must
# be running (the `worker` docker-compose service) for queued imports
# to actually happen -- queuing a job never fails just because no
# worker is currently listening (RabbitMQ holds it durably), but
# nothing processes it until a worker is up.
RABBITMQ_URL: str = os.environ.get("RABBITMQ_URL", "amqp://guest:guest@localhost:5673/")

PRODUCT_IMPORT_QUEUE_NAME: str = "product_import"

# Caps how many products one POST /products/import call can enqueue,
# since the endpoint reads every file into memory before returning --
# an unbounded batch is an easy way to exhaust the API process's memory.
MAX_BULK_IMPORT_ITEMS: int = int(os.environ.get("MAX_BULK_IMPORT_ITEMS", "100"))

# --- Personalized recommendations --------------------------------------------
# A storefront (e.g. the Magento connector) reports what shoppers do via
# POST /events; GET /recommendations turns one shopper's recent history
# into a "taste vector" (a decayed, weighted average of the embeddings of
# the products they interacted with) and returns the nearest products.
# See docs/architecture.md's "Personalized recommendations" section.
#
# How strongly each signal pulls the taste vector: buying says far more
# about taste than glancing at a product page.
EVENT_WEIGHTS: dict[str, float] = {"view": 1.0, "add_to_cart": 3.0, "purchase": 5.0}

# An event's weight halves every this-many days, so recent behavior
# dominates and old interests fade rather than sticking forever.
RECOMMEND_HALF_LIFE_DAYS: float = float(os.environ.get("RECOMMEND_HALF_LIFE_DAYS", "14"))

# Events older than this are ignored entirely (also bounds the popularity fallback).
RECOMMEND_WINDOW_DAYS: int = int(os.environ.get("RECOMMEND_WINDOW_DAYS", "90"))

# Only a shopper's N most recent events feed the taste vector -- bounds
# per-request work no matter how much history a heavy shopper has.
RECOMMEND_MAX_HISTORY: int = int(os.environ.get("RECOMMEND_MAX_HISTORY", "50"))

RECOMMEND_DEFAULT_LIMIT: int = 6
RECOMMEND_MAX_LIMIT: int = 20

# Events accepted per POST /events call.
MAX_EVENTS_PER_REQUEST: int = 100

# Opaque shopper identifier chosen by the client (e.g. "c42" for customer
# 42, "g<random>" for a guest cookie). Deliberately not an email/name: the
# service never needs to know who a shopper *is*, only which events belong
# together. Also keeps the id safe to use in a URL path.
SHOPPER_ID_PATTERN: str = r"^[A-Za-z0-9_-]{1,64}$"

# Storefront traffic all arrives from one client IP (the shop's server),
# so these are far higher than the per-user budgets on /search.
EVENTS_RATE_LIMIT: str = os.environ.get("EVENTS_RATE_LIMIT", "600/minute")
RECOMMENDATIONS_RATE_LIMIT: str = os.environ.get("RECOMMENDATIONS_RATE_LIMIT", "300/minute")

# --- Search configuration ---------------------------------------------------
TOP_K_RESULTS: int = 5

# Files considered a valid "product photo" inside a catalog/<sku>/ folder.
SUPPORTED_IMAGE_NAMES: tuple[str, ...] = ("image.jpg", "image.jpeg", "image.png")

METADATA_FILENAME: str = "metadata.json"

# A new product's `sku` becomes a literal catalog/<sku>/ folder name
# (see IndexingService.add_product), so this doubles as a path-traversal
# guard, not just a style rule: lowercase letters/digits/hyphens only,
# no leading/trailing hyphen, no "..", "/", or "\".
SKU_PATTERN: str = r"^[a-z0-9]+(?:-[a-z0-9]+)*$"
