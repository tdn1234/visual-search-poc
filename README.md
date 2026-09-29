# Visual Product Search — Local POC

Upload a product photo, get back the top 5 most visually similar
products from a small local catalog. Runs entirely on your laptop:
no cloud APIs, no paid services (Postgres runs locally too, via Docker).

## How it works (short version)

```
Product image → CLIP embedding → (optional) color adapter → Postgres/pgvector → cosine similarity → API response
```

See [`docs/architecture.md`](docs/architecture.md) for the full explanation.

By default this ranks purely on CLIP's overall visual/semantic
similarity, which doesn't weight color highly — a red shoe query can
rank other-colored shoes above same-colored other items. The optional
**color adapter** (see [below](#color-adapter-optional-local-fine-tune))
fixes that with a small trained head, without touching CLIP itself.

There are also two ways to filter results by category/color instead of
just ranking by overall similarity — a plain metadata filter
([`GET /products`](#get-products-plain-metadata-filter)) and checkboxes
on the image search itself
([`match_category`/`match_color`](#matching-the-uploaded-images-own-categorycolor)).

Every endpoint except `/health` requires an API key and is rate-limited
— see [Authentication & rate limiting](#authentication--rate-limiting).

## Quick start

Two ways to run this. Docker is the least fiddly (no local Python/venv
setup), the local option is faster to iterate on if you're editing code.

| | Docker | Local Python |
|---|---|---|
| Setup effort | Low — just Docker | Medium — venv + system Python 3.12 |
| First run | Slower (builds image + downloads CLIP) | Slower (installs deps + downloads CLIP) |
| Best for | Just trying it out | Actively developing the code |

Jump to [Option A: Docker](#option-a-docker-recommended-for-first-run) or
[Option B: Local Python setup](#option-b-local-python-setup).

## Project structure

```
visual-search-poc/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI app + startup wiring
│   │   ├── config.py                # paths, model name, constants
│   │   ├── models/
│   │   │   ├── clip_model.py        # CLIP model wrapper (image/text -> vector, zero-shot classify)
│   │   │   ├── color_adapter.py     # optional color-aware head + loader
│   │   │   └── category_classifier.py  # trained category-classification head + loader
│   │   ├── services/
│   │   │   ├── embedding_service.py     # bytes/file -> (adapted) embedding
│   │   │   ├── indexing_service.py      # write path: whole-catalog reindex + single-product add
│   │   │   ├── product_query_service.py # read path: pgvector search + metadata filter, per request
│   │   │   └── attribute_classifier_service.py  # classify an uploaded image's category/color
│   │   ├── api/
│   │   │   ├── search.py            # POST /search route
│   │   │   └── products.py          # GET /products + POST /products (add a product) routes
│   │   ├── db.py                    # Postgres connection pool + pgvector schema setup
│   │   ├── auth.py                  # X-API-Key header check (require_api_key)
│   │   ├── rate_limit.py            # Redis-backed slowapi Limiter instance
│   │   ├── logging_config.py        # log format + request-ID correlation filter
│   │   ├── middleware.py            # assigns request IDs, logs one access-log line per request
│   │   └── schemas/
│   │       └── search.py            # Pydantic request/response models
│   ├── scripts/
│   │   ├── build_embeddings.py      # CLI: catalog/ -> Postgres `products` table
│   │   ├── generate_training_data.py # CLI: synthetic color/category dataset
│   │   ├── train_color_adapter.py    # CLI: train the color adapter head
│   │   ├── train_category_classifier.py  # CLI: train the category classifier head
│   │   └── import_real_photos.py     # CLI: merge real_training/ photos into the manifest
│   ├── data/
│   │   ├── color_adapter.pt         # trained adapter weights (optional, generated)
│   │   ├── category_classifier.pt   # trained classifier weights (optional, generated)
│   │   ├── training/                # synthetic + imported real training set (generated)
│   │   └── real_training/           # your real photos to import (gitignored, you provide)
│   ├── requirements.txt
│   └── Dockerfile
├── docker-compose.yml
├── catalog/
│   ├── shoe-red/
│   │   ├── image.jpg
│   │   └── metadata.json
│   └── ... (17 sample products included)
├── docs/
│   ├── architecture.md
│   └── sample_embeddings.json       # illustrative format only
└── README.md
```

A sample catalog of 17 generated placeholder products is already
included under `catalog/` so you can run the whole pipeline
immediately — it spans 4 categories (Shoes, Bags, Apparel,
Accessories) and 8 colors, with enough overlap
(e.g. red shoes *and* red hats *and* red t-shirts) to actually see the
color adapter's effect. Swap in real product photos whenever you're
ready — just keep the same `<sku>/image.jpg` + `<sku>/metadata.json`
layout.

## Product catalog format

```
catalog/
  shoe-red/
    image.jpg
    metadata.json
  shoe-blue/
    image.jpg
    metadata.json
```

`metadata.json`:

```json
{
  "sku": "shoe-red",
  "name": "Running Shoe Red",
  "price": 99,
  "category": "Shoes",
  "color": "red"
}
```

Required fields: `sku`, `name`, `price`, `category`. `color` is
optional. Two independent things use `category`/`color`: the
[color adapter](#color-adapter-optional-local-fine-tune) nudges
*ranking* based on the image itself (no metadata involved), while
[`GET /products`](#get-products-plain-metadata-filter) and the
[`match_category`/`match_color`](#matching-the-uploaded-images-own-categorycolor)
search checkboxes filter against this metadata field directly. The
image must be named `image.jpg`, `image.jpeg`, or `image.png`.

## Product index storage (Postgres + pgvector)

The product index lives in a `products` table in Postgres, created
automatically at startup (`app/db.py`) using the
[`pgvector`](https://github.com/pgvector/pgvector) extension:

```sql
CREATE TABLE products (
    sku TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    price DOUBLE PRECISION NOT NULL,
    category TEXT NOT NULL,
    color TEXT,
    image_path TEXT NOT NULL,
    embedding VECTOR(512) NOT NULL
);
CREATE INDEX products_embedding_idx ON products USING hnsw (embedding vector_cosine_ops);
```

`embedding` is 512-dimensional and L2-normalized. It's
`clip-vit-base-patch32`'s raw image-encoder output **unless** a
trained color adapter is present (`backend/data/color_adapter.pt`), in
which case it's that output passed through the adapter — see
[below](#color-adapter-optional-local-fine-tune). Either way, every
stored embedding and every query embedding at search time go through
the exact same transformation (`EmbeddingService`), so they stay
comparable. See
[`docs/sample_embeddings.json`](docs/sample_embeddings.json) for an
illustrative shape of one row (truncated for readability — real
vectors have all 512 values).

The table is **generated**, never hand-edited — run
`build_embeddings.py` to (re)build it. Each run `TRUNCATE`s and
re-inserts the whole catalog, so it's safe to re-run any time products,
photos, or the color adapter change. Ranking itself happens **in SQL**:
`ProductQueryService.search_similar` runs one query per request
(`... ORDER BY embedding <=> $query LIMIT 5`), which pgvector answers
using the `products_embedding_idx` HNSW index -- the catalog is never
loaded into the API process at all, so this scales to a catalog far
bigger than would fit comfortably in memory without any change to the
Python code above it.

## Option A: Docker (recommended for first run)

> Requires Docker + the Compose plugin (`docker compose ...`) or the
> standalone `docker-compose` binary (`docker-compose ...` — used
> below; swap in whichever your machine has).
>
> All commands below assume you're inside `visual-search-poc/` (the
> project root, where `docker-compose.yml` lives).

`docker-compose.yml` defines four services: `db` (`pgvector/pgvector:pg16`,
the product index), `adminer` (a lightweight DB dashboard for poking at
`db`), `redis` (backs the request rate limiter), and `backend` (this
API). The API is exposed on **host port 8010** (mapped to port 8000
inside the container) so it doesn't clash with anything already using
8000 on your machine; Postgres is exposed on **host port 5433**
(mapped to 5432) so it doesn't clash with a Postgres you might already
have running locally; Adminer is exposed on **host port 8081**; Redis
is exposed on **host port 6380** (mapped to 6379), same reasoning as
Postgres. Edit the `ports:` lines in `docker-compose.yml` if you want
different host ports.

### 1. Build the image

```bash
docker-compose build
```

This installs Python deps (`torch`, `transformers`, `psycopg`, etc.)
into the `backend` image. First build downloads a few hundred MB and
can take several minutes; it's cached after that. The `db` image is
pulled, not built.

### 2. Start Postgres

```bash
docker-compose up -d db
```

Waits until healthy (`pg_isready`); `backend` also declares this as a
`depends_on` health condition, so `docker-compose up` (step 4) would
wait for it anyway — this step just lets you build the index (step 3)
before starting the API. Data persists in the named volume `pgdata`
across restarts and `docker-compose down` (not `down -v`).

### 3. Build the embeddings index

Run the indexing script as a one-off container using the same image
(no need to start the API first):

```bash
docker-compose run --rm backend python scripts/build_embeddings.py
```

This computes embeddings and writes them into the `products` table in
the `db` service (replacing whatever was there before), and downloads
CLIP's weights (~600 MB, once) into a named Docker volume
(`huggingface_cache`) so later runs/rebuilds don't re-download them.

### 4. Start the API

```bash
docker-compose up
```

The API is now live at `http://127.0.0.1:8010`. Interactive docs
(Swagger UI) are at `http://127.0.0.1:8010/docs`. Stop it with `Ctrl+C`,
or run detached with `docker-compose up -d` and stop later with
`docker-compose down`.

`app/`, `scripts/`, `catalog/`, and `backend/data/` are all
volume-mounted into the container and `uvicorn` runs with `--reload`,
so editing code or the catalog on your host is picked up without
rebuilding the image. You only need to `docker-compose build` again if
you change `backend/requirements.txt` or the `Dockerfile`.

### Browsing the database (Adminer)

[Adminer](https://www.adminer.org/) is a single-file DB dashboard —
much lighter than pgAdmin (no separate login/config volume, starts
instantly) — useful for eyeballing the `products` table without
reaching for `psql`. It comes up automatically with `docker-compose up`
(or start it alone: `docker-compose up -d adminer`, which also brings
up `db` since it depends on it).

Open `http://127.0.0.1:8081` and log in with:

| Field | Value |
|---|---|
| System | PostgreSQL |
| Server | `db` (pre-filled) |
| Username | `postgres` |
| Password | `postgres` |
| Database | `visual_search` |

It's dev-only — no auth beyond the DB credentials above, and not meant
to be exposed beyond localhost. Everything actually stored is
non-sensitive (product metadata + embedding vectors), but don't publish
port 8081 to the open internet as-is.

### Docker troubleshooting

| Symptom | Fix |
|---|---|
| `port is already allocated` | Something else on your host is using port 8010, 5433, 8081, or 6380. Change the host-side port in `docker-compose.yml`'s `ports:` (e.g. `"8020:8000"`, `"5434:5432"`, `"8082:8080"`, or `"6381:6379"`). |
| Build hangs/times out downloading torch | Slow network — just retry `docker-compose build`; pip resumes from cache where possible. |
| `backend` exits/restarts immediately, logs show a Postgres connection error | `db` isn't healthy yet — `docker-compose up -d db` first and wait for `docker-compose ps` to show `(healthy)`, or just re-run `docker-compose up` (the `depends_on` health check should handle this automatically). |
| `/search` returns 503 "Catalog not indexed yet" | You skipped step 3, or the `products` table is empty — run the `build_embeddings.py` one-off command above. |

## Option B: Local Python setup

> Requires Python 3.12 (3.10+ also works), a reachable Postgres with
> the `pgvector` extension available, and a reachable Redis. All
> commands below assume you're inside `visual-search-poc/backend/`.

### 0. Start Postgres and Redis

Easiest path even for "local" development: let Docker run just the
`db` and `redis` services (from the project root, `visual-search-poc/`),
and run everything else natively:

```bash
docker-compose up -d db redis
```

These listen on `localhost:5433` and `localhost:6380`, which are
`app/config.py`'s defaults for `DATABASE_URL`/`REDIS_URL` — no extra
setup needed. Point at different instances by setting those env vars
yourself, e.g.:

```bash
export DATABASE_URL="postgresql://user:pass@localhost:5432/visual_search"
export REDIS_URL="redis://localhost:6379/0"
```

(Postgres must have the `vector` extension installed — the
`pgvector/pgvector` Docker image already includes it; a self-managed
Postgres needs `CREATE EXTENSION vector` permissions and the
extension's files present, see
[pgvector's install docs](https://github.com/pgvector/pgvector#installation)).

You'll also want to set `API_KEY` (see
[Authentication & rate limiting](#authentication--rate-limiting)) —
otherwise the app falls back to the default dev key and logs a warning:

```bash
export API_KEY="some-key-only-you-know"
```

### 1. Create and activate a virtual environment

```bash
cd visual-search-poc/backend
python3.12 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install --upgrade pip

# Install the CPU-only torch build FIRST. Skipping this step makes
# pip fall back to PyPI's default GPU build, which drags in ~2GB of
# unneeded NVIDIA CUDA libraries.
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
```

> The first `transformers`/`torch` install can take a few minutes.
> On the very first run of the app or the build script, CLIP's
> weights (~600 MB) are downloaded once from Hugging Face and cached
> locally (`~/.cache/huggingface`) — subsequent runs are fast and
> fully offline.

### 3. Build the embeddings index

```bash
python scripts/build_embeddings.py
```

Expected output (abridged):

```
INFO ... Catalog directory: .../visual-search-poc/catalog
INFO ... Connecting to Postgres...
INFO ... Connected to Postgres and ensured the pgvector schema exists.
INFO ... Loading CLIP model (this can take a while on first run...)
INFO ... CLIP model loaded successfully.
INFO ... Indexed product 'bag-black'
INFO ... Indexed product 'bag-blue'
...
INFO ... Wrote 17 product embeddings to Postgres
INFO ... Done. Indexed 17 products in 12.3s.
```

### 4. Start the API

```bash
uvicorn app.main:app --reload
```

The API is now live at `http://127.0.0.1:8000`. Interactive docs
(Swagger UI) are at `http://127.0.0.1:8000/docs`.

## Color adapter (optional local fine-tune)

Plain CLIP similarity ranks on overall visual/semantic similarity, so
it doesn't weight color highly — searching a brown shoe can surface
other-colored shoes above brown non-shoes. Fully fine-tuning CLIP
itself would need a GPU and thousands of examples, which doesn't fit
a laptop POC. Instead, `ColorAdapter`
(`app/models/color_adapter.py`) is a small residual head trained *on
top of* frozen CLIP embeddings — cheap enough to train on CPU in
seconds, once embeddings are precomputed.

This is entirely optional: if `backend/data/color_adapter.pt` doesn't
exist, `EmbeddingService` falls back to raw CLIP embeddings and
everything works as before.

### 1. Generate the synthetic training set

There's no large labeled photo dataset locally, so this generates one:
colored geometric shapes (5 categories × 7 colors × several jittered
variants) in the same visual style as the placeholder catalog, with
captions like `"a red shoe"`.

```bash
python scripts/generate_training_data.py
# Docker: docker-compose run --rm backend python scripts/generate_training_data.py
```

Writes images + a `manifest.json` to `backend/data/training/`.
Re-run anytime to refresh/grow the set.

### 2. Train the adapter

```bash
python scripts/train_color_adapter.py
# Docker: docker-compose run --rm backend python scripts/train_color_adapter.py
```

This precomputes (and caches) frozen CLIP image/text embeddings for
the training set, then trains `ColorAdapter` with a CLIP-style
contrastive loss so an image's adapted embedding moves closer to its
color/category caption. Useful flags: `--epochs`, `--batch-size`,
`--lr`, `--no-cache` (force re-encoding instead of reusing
`embedding_cache.npz`). Saves weights to `backend/data/color_adapter.pt`.

On a CPU laptop, encoding ~600 training images takes roughly a minute
or two (one-time, cached after); training itself is a few seconds.

### 3. Rebuild the index and restart

The adapter only takes effect on embeddings computed *after* it's
loaded, so both the catalog and future queries need to go through it:

```bash
python scripts/build_embeddings.py   # rebuilds the Postgres index using the adapter
uvicorn app.main:app --reload        # or: docker-compose restart backend
```

Startup logs confirm whether it found a checkpoint:

```
INFO ... Loaded color adapter from .../backend/data/color_adapter.pt
```

If you skip training, you'll instead see a warning that it's falling
back to raw CLIP — that's expected and non-fatal.

## Category classifier (powers `match_category`)

Used by the `match_category` search checkbox (below) to figure out
"what category is *in* this uploaded photo." The first version of
this used zero-shot CLIP classification (compare the image to text
prompts like `"a photo of shoes"`) — the same technique that works
fine for color. It didn't work for category: on a real test image,
similarity scores across all four categories landed within 0.02 of
each other, essentially random, because this catalog's placeholder
images are abstract solid-color shapes that pretrained CLIP was never
exposed to as "product categories." Color survives zero-shot because
it's a literal pixel property; category doesn't.

So category classification uses the same fix as color: freeze CLIP,
train a small head (`CategoryClassifier`,
`app/models/category_classifier.py` — a linear layer, simpler than the
color adapter's residual MLP since this is plain classification, not
embedding-space nudging) on the synthetic training set.

```bash
python scripts/generate_training_data.py     # if you haven't already
python scripts/train_category_classifier.py
# Docker: docker-compose run --rm backend python scripts/train_category_classifier.py
```

Reuses the same cached embeddings as `train_color_adapter.py`
(`embedding_cache.npz`) — training itself takes well under a second.
Saves weights to `backend/data/category_classifier.pt`; `main.py`
loads it at startup the same way it loads the color adapter, and
`AttributeClassifierService` falls back to (unreliable) zero-shot
classification with a warning if no checkpoint exists.

**On keeping the training set's shapes matched to the real catalog:**
`generate_training_data.py`'s synthetic shapes need to actually
resemble the real catalog's placeholders, or the classifier trains
well on its own held-out validation split (97%+) but doesn't
generalize to the real catalog it's meant to classify. This bit twice
during development: the training set's hat was a half-circle while
every real hat is a full circle, and its t-shirt had collar notches
while every real t-shirt is a plain rectangle. Both are now aligned
(see `CATEGORY_BOX_SIZE` and the shape-drawing functions in
`generate_training_data.py`). If you add a new category to the catalog
with a visibly different placeholder style, keep the training
generator's shape in sync or expect the classifier to be unreliable
for it. Current accuracy on this project's real 17-item catalog: 15/17
(the 2 misses are `Bags` vs. `Apparel`, whose placeholder shapes are
now a near-identical rounded-rect vs. sharp-rect at the same size — a
genuinely subtle cue, not a training bug).

**Trained only on synthetic shapes, it doesn't generalize to real
photos.** Tested against a real (fairly messy — bare foot, patterned
background, two objects in frame) photo of a sneaker: the trained
classifier gave "Accessories" 0.307 vs. "Shoes" 0.279 — essentially a
coin flip, and wrong. Plain *zero-shot* CLIP (no training at all) got
the same photo right (0.277 for "Shoes", highest). That's the mirror
image of the earlier synthetic-image problem: the classifier's
decision boundary was fit entirely to the synthetic embedding
distribution, so a real photo lands somewhere it was never calibrated
for, while zero-shot CLIP -- pretrained on millions of real photos --
is actually in its element there. See
[Training with real photos](#training-with-real-photos) below for the
fix.

## Training with real photos

`generate_training_data.py`'s synthetic shapes only teach the category
classifier about *this project's own placeholder catalog*. To make
`match_category` reliable on real product photos, it needs real
examples in the training set too -- `scripts/import_real_photos.py`
merges them into the same manifest the training scripts already read,
so nothing else about the training pipeline changes.

### 1. Organize your photos by category

```
backend/data/real_training/
  Shoes/
    my-sneaker.jpg
    red-another-shoe.jpg
  Bags/
    ...
  Accessories/
    ...
  Apparel/
    ...
```

Folder names must exactly match a real catalog category (`Shoes`,
`Bags`, `Accessories`, `Apparel`). Optionally prefix a filename with a
recognized color and a hyphen (e.g. `red-my-sneaker.jpg`) to also
label its color for color-adapter training -- otherwise the photo
still counts for category training, just not color training. This
directory is gitignored (personal/local photos, not committed).

### 2. Import and retrain

```bash
python scripts/generate_training_data.py   # if you haven't already
python scripts/import_real_photos.py
python scripts/train_category_classifier.py
# Docker: docker-compose run --rm backend python scripts/{import_real_photos,train_category_classifier}.py
```

`import_real_photos.py` is idempotent -- re-run it anytime after
adding more photos; it only imports files not already in the manifest.
It logs each imported photo with its detected category/color.

### 3. Rebuild the index and restart, same as any other retrain

```bash
python scripts/build_embeddings.py
uvicorn app.main:app --reload   # or: docker-compose restart backend
```

### A real but limited result so far

Adding just **one** real photo (a genuinely messy real-world shot, not
a clean product photo) flipped that exact photo's `match_category`
prediction from wrong ("Accessories") to correct ("Shoes"). That
proves the import → retrain → predict pipeline works end to end, but
**not** that the classifier now generalizes to real photos in
general -- with a single example, "learned to recognize shoes" and
"memorized this one photo" are indistinguishable. A meaningful
generalization check needs either a held-out real photo not used in
training, or enough real examples per category (roughly a handful or
more) that training/validation on real data actually means something.
Treat `match_category` as unproven on new real photos until you've
done one of those.

## Filtering by category/color

Two independent ways to narrow results, on top of everything above:

### `GET /products` (plain metadata filter)

No image involved — just filters the catalog's `category`/`color`
metadata fields, exact and case-insensitive:

```bash
curl -H "X-API-Key: dev-api-key-change-me" \
  "http://127.0.0.1:8010/products?category=Shoes&color=red"
```

Passing both filters is **AND** (narrows to items matching both — e.g.
this returns only red shoes). Passing neither returns the whole
catalog. An unmatched filter returns `{"results": []}`, not an error.
See `app/api/products.py`.

### Matching the uploaded image's own category/color

`POST /search` takes two optional boolean query params,
`match_category` and `match_color` (they show up as checkboxes in
Swagger UI at `/docs` — booleans render that way automatically). Check
one to have the *uploaded image itself* classified (via
`AttributeClassifierService` — zero-shot for color, the trained
`CategoryClassifier` for category) and results restricted to catalog
items sharing that prediction:

```bash
curl -X POST "http://127.0.0.1:8010/search?match_color=true" \
  -H "X-API-Key: dev-api-key-change-me" \
  -F "file=@catalog/shoe-red/image.jpg;type=image/jpeg"
# every result is guaranteed red

curl -X POST "http://127.0.0.1:8010/search?match_category=true&match_color=true" \
  -H "X-API-Key: dev-api-key-change-me" \
  -F "file=@catalog/shoe-red/image.jpg;type=image/jpeg"
# every result is a shoe, or red, or both
```

Passing both is **OR** (widens rather than narrows — the opposite of
`/products`, deliberately: narrowing to AND here would mean neither
filter alone could ever broaden a search). See the docstring on
`search_by_image` in `app/api/search.py`, and `product_query_service.py`
for where the AND/OR split actually lives (two different `WHERE`
clauses built by `list_products` vs. `search_similar`).

## Adding a product (`POST /products`)

Adds one new product: its metadata plus a photo, which gets embedded
with the exact same `EmbeddingService` transformation `POST /search`
uses for query images (through the color adapter too, if one's
loaded) — so the new product is searchable immediately, no separate
reindex step needed.

```bash
curl -X POST "http://127.0.0.1:8010/products" \
  -H "X-API-Key: dev-api-key-change-me" \
  -F "sku=bag-purple" \
  -F "name=Canvas Bag Purple" \
  -F "price=59.99" \
  -F "category=Bags" \
  -F "color=purple" \
  -F "file=@/path/to/photo.jpg;type=image/jpeg"
```

```json
{"sku":"bag-purple","name":"Canvas Bag Purple","price":59.99,"category":"Bags","color":"purple"}
```

| Field | Required | Notes |
|---|---|---|
| `sku` | yes | Unique. Lowercase letters/digits/hyphens only (e.g. `bag-purple`) — becomes a literal `catalog/<sku>/` folder name, so this is a path-safety rule, not just style. |
| `name` | yes | |
| `price` | yes | Must be > 0. |
| `category` | yes | |
| `color` | no | |
| `file` | yes | JPEG or PNG only (not WEBP, even though `/search` accepts WEBP for *query* images — see below). |

**Why this also writes to `catalog/<sku>/`, not just the database:**
`scripts/build_embeddings.py` rebuilds the whole `products` table from
a scan of `catalog/` and **replaces every row**. If a product added
through this endpoint only existed in the database, the next routine
reindex would silently delete it. So `IndexingService.add_product`
writes `catalog/<sku>/image.{jpg,png}` + `catalog/<sku>/metadata.json`
(the same layout as every other product) *and* inserts the DB row —
the two stay in sync, and a future reindex picks this product back up
instead of dropping it. This is also why uploads are restricted to
JPEG/PNG: `SUPPORTED_IMAGE_NAMES` (what the reindex scan recognizes)
doesn't include `.webp`.

**Errors:**

| Situation | HTTP status |
|---|---|
| `sku` already exists | 409 |
| Invalid `sku` format, non-positive `price`, or a missing required field | 422 |
| Unsupported/missing/corrupt image | 400 |
| Missing/invalid `X-API-Key` | 401 |
| Rate limit exceeded (`CREATE_PRODUCT_RATE_LIMIT`, default 10/minute — tighter than `GET /products`' 60/minute, since this runs CLIP inference and writes to disk) | 429 |

On any failure *after* the catalog folder was created (e.g. the DB
insert fails), `IndexingService.add_product` removes that folder again
before returning the error — a failed request never leaves a
half-written product on disk.

## Testing the API

> Use port `8010` if you started the API via Docker (Option A), or
> `8000` if you started it locally (Option B).

### curl

```bash
curl -X POST "http://127.0.0.1:8000/search" \
  -H "accept: application/json" \
  -H "X-API-Key: dev-api-key-change-me" \
  -F "file=@catalog/shoe-red/image.jpg;type=image/jpeg"
# Docker users: replace 8000 with 8010
```

### Postman

1. Method: `POST`
2. URL: `http://127.0.0.1:8000/search` (or `:8010` for Docker)
3. Headers → `X-API-Key: dev-api-key-change-me` (or whatever you set `API_KEY` to)
4. Body → `form-data`
5. Key: `file`, type `File`, value: pick any product photo
6. Send

### Expected response

This is real output from querying with `catalog/shoe-red/image.jpg`
against the included sample catalog, **with a trained color adapter
applied** (see [above](#color-adapter-optional-local-fine-tune)):

```json
{
  "results": [
    { "sku": "shoe-red",   "name": "Running Shoe Red",   "price": 99.0,  "category": "Shoes",       "score": 1.0 },
    { "sku": "tshirt-red", "name": "Cotton T-Shirt Red", "price": 25.0,  "category": "Apparel",     "score": 0.5194 },
    { "sku": "hat-red",    "name": "Baseball Cap Red",   "price": 19.0,  "category": "Accessories", "score": 0.5176 },
    { "sku": "shoe-black", "name": "Running Shoe Black", "price": 109.0, "category": "Shoes",       "score": 0.3955 },
    { "sku": "shoe-blue",  "name": "Running Shoe Blue",  "price": 99.0,  "category": "Shoes",       "score": 0.349 }
  ]
}
```

The exact query image scores 1.0 (identical), and — because of the
color adapter — other **red** items (`tshirt-red`, `hat-red`) outrank
other **shoes** (`shoe-black`, `shoe-blue`). Without the adapter (raw
CLIP, `backend/data/color_adapter.pt` absent/not loaded), the same
query ranks the other shoes highest instead:

```json
{
  "results": [
    { "sku": "shoe-red",   "score": 1.0 },
    { "sku": "shoe-black", "score": 0.9092 },
    { "sku": "shoe-blue",  "score": 0.9054 },
    { "sku": "tshirt-red", "score": 0.8614 },
    { "sku": "hat-red",    "score": 0.8344 }
  ]
}
```

i.e. plain CLIP similarity weighs shape/category more than color; the
adapter flips that. Scores will differ with your own photos — the
included sample catalog uses simple generated placeholder shapes, not
real product photos.

### Health check

```bash
curl http://127.0.0.1:8000/health   # or :8010 for Docker
# {"status":"ok"}
```

`/health` is the one endpoint that needs no API key and isn't
rate-limited — it's what `docker-compose`/an orchestrator would poll,
and gating that on the same key that guards real data would make
liveness checks a secret-management problem for no benefit.

## Authentication & rate limiting

Every route except `/health` requires an `X-API-Key` header
(`app/auth.py`) and is rate-limited per client IP via Redis
(`app/rate_limit.py`, using [`slowapi`](https://github.com/laurentS/slowapi)):

| Endpoint | Default limit | Env var to override | Why |
|---|---|---|---|
| `POST /search` | 20/minute | `SEARCH_RATE_LIMIT` | Runs CLIP inference (the expensive part) plus a DB query. |
| `GET /products` | 60/minute | `PRODUCTS_RATE_LIMIT` | Plain SQL filter, no ML involved — cheap enough for a looser limit. |
| `POST /products` | 10/minute | `CREATE_PRODUCT_RATE_LIMIT` | Runs CLIP inference *and* writes to disk — the tightest budget of the three. |

All three live in `app/config.py` (env-overridable, same pattern as
`API_KEY`/`DATABASE_URL`), in [`limits`-library syntax](https://limits.readthedocs.io/en/stable/quickstart.html#rate-limit-string-notation)
(`"<count>/<second|minute|hour|day>"`). To change one without touching
code:

```bash
# docker-compose: uncomment/edit the matching line in docker-compose.yml, then
docker-compose up -d backend

# local Python:
export SEARCH_RATE_LIMIT="50/minute"
```

**Setting the key:** `docker-compose.yml` sets `API_KEY=dev-api-key-change-me`
for local use. **Change it** before running this anywhere reachable by
anyone but you — `main.py` logs a startup warning if it detects the
default is still in use. For local Python (Option B), set it yourself:

```bash
export API_KEY="some-key-only-you-know"
```

**Sending it:** every `/search`/`/products` call needs the header:

```bash
curl -H "X-API-Key: <your key>" "http://127.0.0.1:8010/products"
```

Swagger UI at `/docs` also has an **Authorize** button (top right) —
paste the key in once and every "Try it out" call in the browser sends
it automatically.

**Missing/wrong key** → `401 Unauthorized`:

```json
{"detail":"Missing or invalid API key. Send it as the 'X-API-Key' header."}
```

**Over the limit** → `429 Too Many Requests`:

```json
{"error":"Rate limit exceeded: 20 per 1 minute"}
```

Limits are counted **per client IP**, not per API key — this POC has
exactly one shared key, so keying by it would put every caller in the
same bucket instead of limiting each one individually. Counters live
in Redis with a self-expiring TTL matching the window (a minute), so
nothing needs manual cleanup; restarting the `redis` container/service
simply resets everyone's count to zero.

## Logging and request correlation

Every request gets a short ID (from an inbound `X-Request-ID` header,
or a freshly generated one) that's attached to every log line emitted
while handling it, and echoed back as an `X-Request-ID` response
header. This is what makes `docker-compose logs` useful for debugging
one specific request instead of scrolling through everything:

```bash
docker-compose logs backend | grep 8ab05fc3a21b
```

Every route logs one line on completion:

```
2026-09-29 08:26:56 [INFO] [8ab05fc3a21b] app.middleware: GET /products -> 200 (7.4ms)
```

**Set `LOG_LEVEL=DEBUG`** (env var, `app/config.py`) to additionally
log a per-step timing breakdown from the services that do the real
work -- CLIP embedding, attribute classification, the pgvector query
-- so a slow `/search` call is diagnosable from the logs alone,
without adding print statements or attaching a profiler:

```bash
# docker-compose: uncomment the LOG_LEVEL line in docker-compose.yml, then
docker-compose up -d backend

# local Python:
export LOG_LEVEL=DEBUG
```

```
2026-09-29 08:30:39 [DEBUG] [16d1...] app.services.embedding_service: CLIP embed (upload, JPEG) took 752.9ms
2026-09-29 08:30:39 [DEBUG] [16d1...] app.services.attribute_classifier_service: Attribute classification took 686.0ms (category='Shoes', color='red')
2026-09-29 08:30:39 [DEBUG] [16d1...] app.services.product_query_service: pgvector search_similar (top_k=5, category='Shoes', color='red') took 4.4ms, 5 rows
2026-09-29 08:30:39 [INFO]  [16d1...] app.middleware: POST /search -> 200 (830.2ms)
```

(real output, from a `match_category=true&match_color=true` search —
shows attribute classification, not CLIP embedding or the DB query, as
the dominant cost when those checkboxes are on.) Third-party libraries'
own DEBUG logs (`multipart`, `urllib3`, `PIL`, `httpcore`/`httpx`) are
pinned to `WARNING` regardless of `LOG_LEVEL`, since their protocol-level
internals are noise here, not debugging signal — see
`app/logging_config.py`'s `_NOISY_LOGGERS`.

## Error handling

| Situation | HTTP status | Detail |
|---|---|---|
| Missing/invalid `X-API-Key` header | 401 | Missing or invalid API key |
| Rate limit exceeded | 429 | Rate limit exceeded: `<limit>` |
| Non-image file uploaded | 400 | Unsupported file type |
| Empty file uploaded | 400 | Uploaded file is empty |
| Corrupted/unreadable image | 400 | Could not encode image with CLIP |
| `products` table missing or empty (`/search`) | 503 | Catalog not indexed yet — run the build script |
| `sku` already exists (`POST /products`) | 409 | Product '\<sku\>' already exists |
| Invalid `sku`/`price`/missing field (`POST /products`) | 422 | Pydantic validation error detail |
| Unexpected server error | 500 | Internal error while searching / creating the product |

## Future improvements

- **V1:** CLIP (`clip-vit-base-patch32`) + JSON file storage + brute-force
  cosine similarity, with an optional trained
  [color adapter](#color-adapter-optional-local-fine-tune) for ranking
  and a trained [category classifier](#category-classifier-powers-match_category)
  + [category/color filters](#filtering-by-categorycolor) for narrowing
  results.
- **V2 (this POC):** CLIP + PostgreSQL with the `pgvector` extension
  (see [Product index storage](#product-index-storage-postgres--pgvector)).
  Replaces the JSON file with a `products` table and an HNSW ANN index,
  run as a `db` service in `docker-compose.yml`. Writes go through
  `IndexingService` (offline, via `build_embeddings.py`); reads go
  through `ProductQueryService`, which runs an actual SQL query per
  request (`ORDER BY embedding <=> $query LIMIT k`, answered by the
  HNSW index) -- the catalog is never loaded into the API process.
- **V3:** Magento 2 module integration. A Magento observer/cron pushes
  product images to this service on save; a Magento block/API calls
  `/search` from the storefront (e.g. a "search by image" widget) and
  resolves returned SKUs back to real Magento product pages.
- **V4:** Free-text query support. `match_category`/`match_color`
  already cover structured filtering; a further step would let a
  query combine an uploaded image with free text (e.g. "under $100")
  via CLIP's text encoder plus structured metadata like `price`.
- **V5:** Recommendation engine. Reuse the same embedding space for
  "customers who viewed this also liked" style recommendations,
  combined with behavioral signals (views, purchases) rather than
  visual similarity alone.

## Notes on code quality

- Every function has type hints and a docstring explaining inputs,
  outputs, and error conditions.
- Services are plain classes with constructor-injected dependencies
  (e.g. `EmbeddingService` receives a `ClipModel` instance) so they're
  easy to test in isolation and easy to swap later.
- The CLIP model is loaded exactly once, at FastAPI startup
  (`app/main.py`'s `lifespan`), not per-request.
- Errors are caught at the boundary where they can be meaningfully
  handled (bad image → 400, empty catalog → 503) rather than silently
  swallowed.
