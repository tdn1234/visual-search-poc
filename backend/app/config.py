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
REDIS_URL: str = os.environ.get("REDIS_URL", "redis://localhost:6380/0")

# Per-client-IP limits, in `limits`-library syntax (e.g. "20/minute",
# "100/hour", "5/second"). Applied via `@limiter.limit(...)` on each
# route in api/search.py / api/products.py. Different endpoints get
# different budgets: /search runs CLIP inference (expensive), /products
# is a plain SQL filter (cheap) -- see docs/architecture.md's "Auth and
# rate limiting" section for the reasoning.
SEARCH_RATE_LIMIT: str = os.environ.get("SEARCH_RATE_LIMIT", "20/minute")
PRODUCTS_RATE_LIMIT: str = os.environ.get("PRODUCTS_RATE_LIMIT", "60/minute")

# --- Search configuration ---------------------------------------------------
TOP_K_RESULTS: int = 5

# Files considered a valid "product photo" inside a catalog/<sku>/ folder.
SUPPORTED_IMAGE_NAMES: tuple[str, ...] = ("image.jpg", "image.jpeg", "image.png")

METADATA_FILENAME: str = "metadata.json"
