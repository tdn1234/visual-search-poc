# Architecture — Visual Product Search POC

## Pipeline overview

```
Product image (catalog/<sku>/image.jpg)
        │
        ▼
 CLIP image encoder (openai/clip-vit-base-patch32)
        │  produces a 512-dim vector, L2-normalized
        ▼
 (optional) Color adapter (ColorAdapter, frozen-CLIP residual head)
        │  only applied if backend/data/color_adapter.pt exists
        ▼
 embeddings.json  (sku, name, price, category, color, embedding)
        │
        │   ... at query time ...
        │
 Uploaded image (POST /search)
        │
        ▼
 Same CLIP encoder → same (optional) color adapter → query embedding
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

The color adapter is optional and off by default (no checkpoint =
`EmbeddingService` returns raw CLIP embeddings). See
[Color adapter, step by step](#color-adapter-step-by-step) below for
why it exists and how it's trained.

## Three separate flows

There are deliberately **three independent flows** that never run in
the same request:

1. **Offline adapter training** (`scripts/generate_training_data.py` +
   `scripts/train_color_adapter.py`) — run manually, and only if you
   want color-aware ranking. Generates a synthetic image/caption
   dataset, precomputes frozen CLIP embeddings for it, and trains
   `ColorAdapter` on those cached vectors. Produces
   `backend/data/color_adapter.pt`. Entirely optional; nothing else
   depends on this having been run.

2. **Offline indexing** (`scripts/build_embeddings.py`) — run manually
   whenever the catalog changes (or after (re)training the adapter).
   Scans `catalog/`, embeds every product photo once (through the
   adapter too, if a checkpoint exists), and writes
   `backend/data/embeddings.json`. This is the expensive step (loading
   CLIP, running inference per image), but it happens outside the
   request/response cycle.

3. **Online search** (`POST /search`, served by FastAPI) — the CLIP
   model (and color adapter, if present) is loaded **once at API
   startup** (see `app/main.py`'s `lifespan` handler) and kept in
   memory. Each request only pays for encoding *one* uploaded image
   plus a single cosine-similarity matrix multiply against the (small,
   20-50 row) catalog — both are fast enough to run comfortably on a
   laptop CPU.

Keeping these flows separate is what makes the JSON-file approach
viable: the "database" (embeddings.json) is rebuilt in batch, and the
API only ever reads it. It's also why flow 2 must be re-run after flow
1 changes the adapter checkpoint — otherwise the catalog's stored
embeddings and freshly-adapted query embeddings would be in different
(non-comparable) vector spaces.

## Why embeddings are pre-normalized

`ClipModel.encode_image` L2-normalizes every embedding it produces
(image norm divided by its own L2 norm) before returning it. Once both
vectors in a comparison are unit-length, cosine similarity reduces to
a plain dot product. This doesn't change the correctness of using
`cosine_similarity` (it's mathematically equivalent either way), but
it does mean the stored vectors are ready to compare directly and
consistently, regardless of which layer does the comparison.

`ColorAdapter.forward` preserves this invariant: it adds a learned
residual to the input embedding and then re-normalizes the result, so
adapted embeddings are unit-length too, whether or not a checkpoint is
loaded.

## Layered responsibilities

| Layer | File | Responsibility |
|---|---|---|
| Model | `app/models/clip_model.py` | Owns the HF CLIP model/processor. The *only* place that imports `torch`/`transformers` for inference. Encodes both images and text. |
| Model | `app/models/color_adapter.py` | `ColorAdapter` (small residual head) + `load_color_adapter`, which loads a trained checkpoint or returns `None`. |
| Service | `app/services/embedding_service.py` | Turns raw bytes or a file path into an embedding, applying the color adapter (if loaded) after CLIP. Validates images. |
| Service | `app/services/indexing_service.py` | Knows the catalog folder layout and the embeddings.json schema. Builds and loads the index. |
| Service | `app/services/similarity_service.py` | Pure ranking logic: cosine similarity + top-K sort. No I/O. |
| API | `app/api/search.py` | HTTP concerns only: validates the upload, calls services, maps errors to HTTP status codes. |
| Entrypoint | `app/main.py` | Wires everything together once at startup (`lifespan`), exposes the FastAPI `app`. |
| Script | `scripts/build_embeddings.py` | CLI entrypoint for the offline indexing flow. |
| Script | `scripts/generate_training_data.py` | CLI entrypoint that generates the synthetic color/category training set. |
| Script | `scripts/train_color_adapter.py` | CLI entrypoint for the offline adapter-training flow. |

Each layer only talks to the layer directly below it, so, for example,
swapping the storage format from JSON to PostgreSQL later only
requires changing `IndexingService` — the API and model layers stay
untouched. Likewise, `EmbeddingService` is the *only* place that knows
whether a color adapter is active — everything downstream of it (the
API route, `SimilarityService`) just sees embeddings and has no idea
whether they came from raw CLIP or an adapted space.

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

## Color adapter, step by step

**Why it exists:** CLIP's embedding optimizes for overall visual/
semantic similarity (shape, category, texture, color all entangled),
not any single attribute. So raw cosine similarity can rank a
different-colored item of the same shape above a same-colored item of
a different shape — e.g. querying a red shoe can rank a black shoe
above a red hat, even though the same-color match is arguably the
better hit. Full fine-tuning CLIP to fix this needs a GPU and
thousands of examples, which doesn't fit this POC's "runs on a laptop"
constraint.

**The fix:** freeze CLIP entirely and train a small extra head,
`ColorAdapter`, on top of its (cached) output embeddings. Since CLIP
itself is never touched, training is just matrix ops on 512-dim
vectors — fast enough for a CPU laptop.

1. `scripts/generate_training_data.py` synthesizes a small labeled
   dataset (no real photo dataset is available locally): colored
   geometric shapes across several categories/colors, each with a
   caption like `"a red shoe"`. Written to `backend/data/training/`.
2. `scripts/train_color_adapter.py` loads frozen `ClipModel`, encodes
   every training image (`encode_image`) and caption (`encode_text`)
   once, and caches those vectors (`embedding_cache.npz`) — this is
   the slow part, everything after is fast.
3. It trains `ColorAdapter` (`embedding_dim -> hidden_dim -> embedding_dim`,
   with a residual connection and re-normalization) using a symmetric
   CLIP-style contrastive (InfoNCE) loss: for each batch, an adapted
   image embedding should score highest against its own caption's text
   embedding and lower against every other caption in the batch.
4. Weights are saved to `backend/data/color_adapter.pt` along with the
   `embedding_dim`/`hidden_dim` needed to reconstruct the model.
5. `load_color_adapter` (called from both `main.py` and
   `build_embeddings.py`) loads that checkpoint into an `EmbeddingService`,
   which then runs *every* embedding it produces — catalog and query
   alike — through `ColorAdapter.forward` after CLIP encoding.

**Net effect:** ranking is unchanged in shape (still cosine similarity
in `SimilarityService`, unaware the adapter exists) — only the vector
space the embeddings live in changes, so color differences contribute
more to the resulting score.
