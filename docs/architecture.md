# Architecture — Visual Product Search POC

## Pipeline overview

```
Product image (catalog/<sku>/image.jpg)
        │
        ▼
 CLIP image encoder (openai/clip-vit-base-patch32)
        │  produces a 512-dim vector, L2-normalized
        ▼
 embeddings.json  (sku, name, price, category, embedding)
        │
        │   ... at query time ...
        │
 Uploaded image (POST /search)
        │
        ▼
 Same CLIP encoder → query embedding (512-dim, normalized)
        │
        ▼
 Cosine similarity vs every embedding in embeddings.json
 (sklearn.metrics.pairwise.cosine_similarity)
        │
        ▼
 Sort descending, take top 5
        │
        ▼
 JSON API response: { "results": [ {sku, name, score}, ... ] }
```

## Two separate flows

There are deliberately **two independent flows** that never run in the
same request:

1. **Offline indexing** (`scripts/build_embeddings.py`) — run manually
   whenever the catalog changes. Scans `catalog/`, embeds every
   product photo once, and writes `backend/data/embeddings.json`.
   This is the expensive step (loading CLIP, running inference per
   image), but it happens outside the request/response cycle.

2. **Online search** (`POST /search`, served by FastAPI) — the CLIP
   model is loaded **once at API startup** (see `app/main.py`'s
   `lifespan` handler) and kept in memory. Each request only pays for
   encoding *one* uploaded image plus a single cosine-similarity
   matrix multiply against the (small, 20-50 row) catalog — both are
   fast enough to run comfortably on a laptop CPU.

Keeping these flows separate is what makes the JSON-file approach
viable: the "database" (embeddings.json) is rebuilt in batch, and the
API only ever reads it.

## Why embeddings are pre-normalized

`ClipModel.encode_image` L2-normalizes every embedding it produces
(image norm divided by its own L2 norm) before returning it. Once both
vectors in a comparison are unit-length, cosine similarity reduces to
a plain dot product. This doesn't change the correctness of using
`cosine_similarity` (it's mathematically equivalent either way), but
it does mean the stored vectors are ready to compare directly and
consistently, regardless of which layer does the comparison.

## Layered responsibilities

| Layer | File | Responsibility |
|---|---|---|
| Model | `app/models/clip_model.py` | Owns the HF CLIP model/processor. The *only* place that imports `torch`/`transformers` for inference. |
| Service | `app/services/embedding_service.py` | Turns raw bytes or a file path into an embedding. Validates images. |
| Service | `app/services/indexing_service.py` | Knows the catalog folder layout and the embeddings.json schema. Builds and loads the index. |
| Service | `app/services/similarity_service.py` | Pure ranking logic: cosine similarity + top-K sort. No I/O. |
| API | `app/api/search.py` | HTTP concerns only: validates the upload, calls services, maps errors to HTTP status codes. |
| Entrypoint | `app/main.py` | Wires everything together once at startup (`lifespan`), exposes the FastAPI `app`. |
| Script | `scripts/build_embeddings.py` | CLI entrypoint for the offline indexing flow. |

Each layer only talks to the layer directly below it, so, for example,
swapping the storage format from JSON to PostgreSQL later only
requires changing `IndexingService` — the API and model layers stay
untouched.

## Similarity ranking, step by step

1. `SimilarityService.rank_products` receives the query embedding and
   the full in-memory catalog (a Python list of `ProductRecord`).
2. It stacks every catalog embedding into one NumPy matrix of shape
   `(num_products, 512)`.
3. `sklearn.metrics.pairwise.cosine_similarity(query, matrix)` returns
   a `(1, num_products)` array of similarity scores, one per product.
4. `np.argsort(-scores)[:top_k]` sorts descending and slices the top 5
   indices.
5. Those indices are mapped back to `SearchResult` objects (sku, name,
   price, category, score) and returned in ranked order.

With 20-50 products this whole computation takes low single-digit
milliseconds — no ANN index (FAISS, HNSW, etc.) is needed at this
scale.
