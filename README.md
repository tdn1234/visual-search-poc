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

There are also two ways to filter results by category/color instead of
just ranking by overall similarity — a plain metadata filter
([`GET /products`](#get-products-plain-metadata-filter)) and checkboxes
on the image search itself
([`match_category`/`match_color`](#matching-the-uploaded-images-own-categorycolor)).

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
│   │   │   ├── similarity_service.py    # cosine similarity + ranking
│   │   │   ├── indexing_service.py      # build/load embeddings.json
│   │   │   ├── attribute_classifier_service.py  # classify an uploaded image's category/color
│   │   │   └── catalog_filter.py        # category/color filtering (AND for /products, OR for /search)
│   │   ├── api/
│   │   │   ├── search.py            # POST /search route
│   │   │   └── products.py          # GET /products route
│   │   └── schemas/
│   │       └── search.py            # Pydantic request/response models
│   ├── scripts/
│   │   ├── build_embeddings.py      # CLI: catalog/ -> embeddings.json
│   │   ├── generate_training_data.py # CLI: synthetic color/category dataset
│   │   ├── train_color_adapter.py    # CLI: train the color adapter head
│   │   └── train_category_classifier.py  # CLI: train the category classifier head
│   ├── data/
│   │   ├── embeddings.json          # generated index (not hand-written)
│   │   ├── color_adapter.pt         # trained adapter weights (optional, generated)
│   │   ├── category_classifier.pt   # trained classifier weights (optional, generated)
│   │   └── training/                # synthetic training set (generated)
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
INFO ... Wrote 17 product embeddings to .../embeddings.json
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
python scripts/build_embeddings.py   # rebuilds embeddings.json using the adapter
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

## Filtering by category/color

Two independent ways to narrow results, on top of everything above:

### `GET /products` (plain metadata filter)

No image involved — just filters the catalog's `category`/`color`
metadata fields, exact and case-insensitive:

```bash
curl "http://127.0.0.1:8010/products?category=Shoes&color=red"
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
  -F "file=@catalog/shoe-red/image.jpg;type=image/jpeg"
# every result is guaranteed red

curl -X POST "http://127.0.0.1:8010/search?match_category=true&match_color=true" \
  -F "file=@catalog/shoe-red/image.jpg;type=image/jpeg"
# every result is a shoe, or red, or both
```

Passing both is **OR** (widens rather than narrows — the opposite of
`/products`, deliberately: narrowing to AND here would mean neither
filter alone could ever broaden a search). See the docstring on
`search_by_image` in `app/api/search.py`, and `catalog_filter.py` for
where the AND/OR split actually lives.

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
  [color adapter](#color-adapter-optional-local-fine-tune) for ranking
  and a trained [category classifier](#category-classifier-powers-match_category)
  + [category/color filters](#filtering-by-categorycolor) for narrowing
  results. Fine up to a few hundred products.
- **V2:** CLIP + PostgreSQL with the `pgvector` extension. Replaces
  `embeddings.json` with a `products` table and an ANN index
  (`ivfflat`/`hnsw`), so `IndexingService`/`SimilarityService` gain
  DB-backed implementations behind the same interfaces.
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
