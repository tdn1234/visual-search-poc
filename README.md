# Visual Product Search — Local POC

Upload a product photo, get back the top 5 most visually similar
products from a small local catalog. Runs entirely on your laptop:
no cloud APIs, no paid services.

## How it works (short version)

```
Product image → CLIP embedding → embeddings.json → cosine similarity → API response
```

See [`docs/architecture.md`](docs/architecture.md) for the full explanation.

## Project structure

```
visual-search-poc/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI app + startup wiring
│   │   ├── config.py                # paths, model name, constants
│   │   ├── models/
│   │   │   └── clip_model.py        # CLIP model wrapper (image -> vector)
│   │   ├── services/
│   │   │   ├── embedding_service.py     # bytes/file -> embedding
│   │   │   ├── similarity_service.py    # cosine similarity + ranking
│   │   │   └── indexing_service.py      # build/load embeddings.json
│   │   ├── api/
│   │   │   └── search.py            # POST /search route
│   │   └── schemas/
│   │       └── search.py            # Pydantic request/response models
│   ├── scripts/
│   │   └── build_embeddings.py      # CLI: catalog/ -> embeddings.json
│   ├── data/
│   │   └── embeddings.json          # generated index (not hand-written)
│   └── requirements.txt
├── catalog/
│   ├── shoe-red/
│   │   ├── image.jpg
│   │   └── metadata.json
│   └── ... (8 sample products included)
├── docs/
│   ├── architecture.md
│   └── sample_embeddings.json       # illustrative format only
└── README.md
```

A sample catalog of 8 generated placeholder products is already
included under `catalog/` so you can run the whole pipeline
immediately. Swap in real product photos whenever you're ready —
just keep the same `<sku>/image.jpg` + `<sku>/metadata.json` layout.

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
  "category": "Shoes"
}
```

Required fields: `sku`, `name`, `price`, `category`. The image must be
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
    "image_path": "catalog/shoe-red/image.jpg",
    "embedding": [0.0182, -0.0431, 0.0705, "... 512 floats total ..."]
  }
]
```

`embedding` is the raw 512-dimensional, L2-normalized output of
`clip-vit-base-patch32`'s image encoder. See
[`docs/sample_embeddings.json`](docs/sample_embeddings.json) for the
exact shape (truncated for readability — real vectors have all 512
values).

This file is **generated**, never hand-edited — run
`build_embeddings.py` to (re)create it.

## Local setup

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
INFO ... Indexed product 'bag-brown'
INFO ... Indexed product 'hat-green'
...
INFO ... Wrote 8 product embeddings to .../embeddings.json
INFO ... Done. Indexed 8 products in 12.3s.
```

### 4. Start the API

```bash
uvicorn app.main:app --reload
```

The API is now live at `http://127.0.0.1:8000`. Interactive docs
(Swagger UI) are at `http://127.0.0.1:8000/docs`.

## Testing the API

### curl

```bash
curl -X POST "http://127.0.0.1:8000/search" \
  -H "accept: application/json" \
  -F "file=@catalog/shoe-red/image.jpg;type=image/jpeg"
```

### Postman

1. Method: `POST`
2. URL: `http://127.0.0.1:8000/search`
3. Body → `form-data`
4. Key: `file`, type `File`, value: pick any product photo
5. Send

### Expected response

```json
{
  "results": [
    { "sku": "shoe-red",  "name": "Running Shoe Red",  "price": 99.0, "category": "Shoes", "score": 0.9994 },
    { "sku": "shoe-blue", "name": "Running Shoe Blue", "price": 99.0, "category": "Shoes", "score": 0.8721 },
    { "sku": "shoe-black","name": "Running Shoe Black","price": 109.0,"category": "Shoes", "score": 0.8544 },
    { "sku": "watch-silver","name": "Classic Watch Silver","price": 149.0,"category": "Accessories","score": 0.5013 },
    { "sku": "hat-green", "name": "Baseball Cap Green", "price": 19.0, "category": "Accessories", "score": 0.4108 }
  ]
}
```

(Exact scores will vary — the included sample catalog uses simple
generated placeholder shapes, not real photos, so treat the relative
ranking as the meaningful part of the demo.)

### Health check

```bash
curl http://127.0.0.1:8000/health
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
  + brute-force cosine similarity. Fine up to a few hundred products.
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
