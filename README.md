# Visual Product Search — Local POC

Upload a product photo, get back the top 5 most visually similar
products from a small local catalog. Runs entirely on your laptop:
no cloud APIs, no paid services.

## How it works (short version)

```
Product image → CLIP embedding → (optional) color adapter → embeddings.json → cosine similarity → API response
```

See [`docs/architecture.md`](docs/architecture.md) for the full explanation.

By default this ranks purely on CLIP's overall visual/semantic
similarity, which doesn't weight color highly — a red shoe query can
rank other-colored shoes above same-colored other items. The optional
**color adapter** (see [below](#color-adapter-optional-local-fine-tune))
fixes that with a small trained head, without touching CLIP itself.

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
│   │   │   ├── clip_model.py        # CLIP model wrapper (image/text -> vector)
│   │   │   └── color_adapter.py     # optional color-aware head + loader
│   │   ├── services/
│   │   │   ├── embedding_service.py     # bytes/file -> (adapted) embedding
│   │   │   ├── similarity_service.py    # cosine similarity + ranking
│   │   │   └── indexing_service.py      # build/load embeddings.json
│   │   ├── api/
│   │   │   └── search.py            # POST /search route
│   │   └── schemas/
│   │       └── search.py            # Pydantic request/response models
│   ├── scripts/
│   │   ├── build_embeddings.py      # CLI: catalog/ -> embeddings.json
│   │   ├── generate_training_data.py # CLI: synthetic color/category dataset
│   │   └── train_color_adapter.py    # CLI: train the color adapter head
│   ├── data/
│   │   ├── embeddings.json          # generated index (not hand-written)
│   │   ├── color_adapter.pt         # trained adapter weights (optional, generated)
│   │   └── training/                # synthetic training set (generated)
│   ├── requirements.txt
│   └── Dockerfile
├── docker-compose.yml
├── catalog/
│   ├── shoe-red/
│   │   ├── image.jpg
│   │   └── metadata.json
│   └── ... (16 sample products included)
├── docs/
│   ├── architecture.md
│   └── sample_embeddings.json       # illustrative format only
└── README.md
```

A sample catalog of 16 generated placeholder products is already
included under `catalog/` so you can run the whole pipeline
immediately — it spans 4 categories (Shoes, Bags, Apparel,
Accessories) and 7 colors, with enough overlap
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
optional — it's stored on `ProductRecord`/`embeddings.json` for future
use (e.g. filtering), but the `/search` response and ranking today
don't read it; color-awareness currently comes entirely from the
[color adapter](#color-adapter-optional-local-fine-tune) acting on the
image embedding, not from this metadata field. The image must be
named `image.jpg`, `image.jpeg`, or `image.png`.

## Embedding storage format

`backend/data/embeddings.json` is a JSON array, one entry per product:

```json
[
  {
    "sku": "shoe-red",
    "name": "Running Shoe Red",
    "price": 99.0,
    "category": "Shoes",
    "color": "red",
    "image_path": "catalog/shoe-red/image.jpg",
    "embedding": [0.0182, -0.0431, 0.0705, "... 512 floats total ..."]
  }
]
```

`embedding` is 512-dimensional and L2-normalized. It's
`clip-vit-base-patch32`'s raw image-encoder output **unless** a
trained color adapter is present (`backend/data/color_adapter.pt`), in
which case it's that output passed through the adapter — see
[below](#color-adapter-optional-local-fine-tune). Either way, every
embedding in this file and every query embedding at search time go
through the exact same transformation (`EmbeddingService`), so they
stay comparable. See
[`docs/sample_embeddings.json`](docs/sample_embeddings.json) for the
exact shape (truncated for readability — real vectors have all 512
values).

This file is **generated**, never hand-edited — run
`build_embeddings.py` to (re)create it.

## Option A: Docker (recommended for first run)

> Requires Docker + the Compose plugin (`docker compose ...`) or the
> standalone `docker-compose` binary (`docker-compose ...` — used
> below; swap in whichever your machine has).
>
> All commands below assume you're inside `visual-search-poc/` (the
> project root, where `docker-compose.yml` lives).

The API is exposed on **host port 8010** (mapped to port 8000 inside
the container) so it doesn't clash with anything already using 8000
on your machine. Edit the `ports:` line in `docker-compose.yml` if you
want a different host port.

### 1. Build the image

```bash
docker-compose build
```

This installs Python deps (`torch`, `transformers`, etc.) into the
image. First build downloads a few hundred MB and can take several
minutes; it's cached after that.

### 2. Build the embeddings index

Run the indexing script as a one-off container using the same image
(no need to start the API first):

```bash
docker-compose run --rm backend python scripts/build_embeddings.py
```

This writes to `./backend/data/embeddings.json` on your host (it's a
mounted volume), and downloads CLIP's weights (~600 MB, once) into a
named Docker volume (`huggingface_cache`) so later runs/rebuilds don't
re-download them.

### 3. Start the API

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

### Docker troubleshooting

| Symptom | Fix |
|---|---|
| `port is already allocated` | Something else on your host is using port 8010. Change the host-side port in `docker-compose.yml`'s `ports:` (e.g. `"8020:8000"`). |
| Build hangs/times out downloading torch | Slow network — just retry `docker-compose build`; pip resumes from cache where possible. |
| `/search` returns 503 "Catalog not indexed yet" | You skipped step 2, or `backend/data/embeddings.json` doesn't exist yet — run the `build_embeddings.py` one-off command above. |

## Option B: Local Python setup

> Requires Python 3.12 (3.10+ also works). All commands below assume
> you're inside `visual-search-poc/backend/`.

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
INFO ... Output file:       .../visual-search-poc/backend/data/embeddings.json
INFO ... Loading CLIP model (this can take a while on first run...)
INFO ... CLIP model loaded successfully.
INFO ... Indexed product 'bag-black'
INFO ... Indexed product 'bag-blue'
...
INFO ... Wrote 16 product embeddings to .../embeddings.json
INFO ... Done. Indexed 16 products in 12.3s.
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
python scripts/build_embeddings.py   # rebuilds embeddings.json using the adapter
uvicorn app.main:app --reload        # or: docker-compose restart backend
```

Startup logs confirm whether it found a checkpoint:

```
INFO ... Loaded color adapter from .../backend/data/color_adapter.pt
```

If you skip training, you'll instead see a warning that it's falling
back to raw CLIP — that's expected and non-fatal.

## Testing the API

> Use port `8010` if you started the API via Docker (Option A), or
> `8000` if you started it locally (Option B).

### curl

```bash
curl -X POST "http://127.0.0.1:8000/search" \
  -H "accept: application/json" \
  -F "file=@catalog/shoe-red/image.jpg;type=image/jpeg"
# Docker users: replace 8000 with 8010
```

### Postman

1. Method: `POST`
2. URL: `http://127.0.0.1:8000/search` (or `:8010` for Docker)
3. Body → `form-data`
4. Key: `file`, type `File`, value: pick any product photo
5. Send

### Expected response

This is real output from querying with `catalog/shoe-red/image.jpg`
against the included sample catalog, **with a trained color adapter
applied** (see [above](#color-adapter-optional-local-fine-tune)):

```json
{
  "results": [
    { "sku": "shoe-red",   "name": "Running Shoe Red",   "price": 99.0,  "category": "Shoes",       "score": 1.0 },
    { "sku": "hat-red",    "name": "Baseball Cap Red",   "price": 19.0,  "category": "Accessories", "score": 0.7695 },
    { "sku": "tshirt-red", "name": "Cotton T-Shirt Red", "price": 25.0,  "category": "Apparel",     "score": 0.5416 },
    { "sku": "hat-black",  "name": "Baseball Cap Black", "price": 19.0,  "category": "Accessories", "score": 0.3599 },
    { "sku": "shoe-black", "name": "Running Shoe Black", "price": 109.0, "category": "Shoes",       "score": 0.2387 }
  ]
}
```

The exact query image scores 1.0 (identical), and — because of the
color adapter — other **red** items (`hat-red`, `tshirt-red`) outrank
other **shoes** (`shoe-black`). Without the adapter (raw CLIP,
`backend/data/color_adapter.pt` absent/not loaded), the same query
ranks the other shoes highest instead:

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

## Error handling

| Situation | HTTP status | Detail |
|---|---|---|
| Non-image file uploaded | 400 | Unsupported file type |
| Empty file uploaded | 400 | Uploaded file is empty |
| Corrupted/unreadable image | 400 | Could not encode image with CLIP |
| `embeddings.json` missing or empty | 503 | Catalog not indexed yet — run the build script |
| Unexpected server error | 500 | Internal error while searching |

## Future improvements

- **V1 (this POC):** CLIP (`clip-vit-base-patch32`) + JSON file storage
  + brute-force cosine similarity, with an optional trained
  [color adapter](#color-adapter-optional-local-fine-tune) on top.
  Fine up to a few hundred products.
- **V2:** CLIP + PostgreSQL with the `pgvector` extension. Replaces
  `embeddings.json` with a `products` table and an ANN index
  (`ivfflat`/`hnsw`), so `IndexingService`/`SimilarityService` gain
  DB-backed implementations behind the same interfaces.
- **V3:** Magento 2 module integration. A Magento observer/cron pushes
  product images to this service on save; a Magento block/API calls
  `/search` from the storefront (e.g. a "search by image" widget) and
  resolves returned SKUs back to real Magento product pages.
- **V4:** Hybrid image + text search. Use CLIP's *text* encoder too,
  so a query can combine an uploaded image with a text filter (e.g.
  "red shoes under $100") by blending/filtering on both embedding
  spaces plus structured metadata.
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
