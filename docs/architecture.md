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
 Postgres `products` table, pgvector VECTOR(512) column
 (sku, name, price, category, color, image_path, embedding)
        │
        │   ... at query time ...
        │
 Uploaded image (POST /search)
        │
        ▼
 Same CLIP encoder → same (optional) color adapter → query embedding
        │
        ▼
 (optional) match_category/match_color: classify the uploaded image
 itself (AttributeClassifierService) to get a predicted category/color
        │
        ▼
 ProductQueryService.search_similar: ONE SQL query --
 `SELECT ... FROM products WHERE (category/color filter) ORDER BY
  embedding <=> $query LIMIT 5` -- pgvector's HNSW index does the
 nearest-neighbor ranking inside Postgres. No catalog ever loaded
 into Python.
        │
        ▼
 JSON API response: { "results": [ {sku, name, score}, ... ] }
```

The color adapter is optional and off by default (no checkpoint =
`EmbeddingService` returns raw CLIP embeddings). See
[Color adapter, step by step](#color-adapter-step-by-step) below for
why it exists and how it's trained. `match_category`/`match_color`
filtering is also optional (off unless the query params are set) --
see [Classifying and filtering an uploaded image](#classifying-and-filtering-an-uploaded-image)
below. `GET /products` is a separate, simpler path: it never touches
CLIP or pgvector at all, just a plain SQL `WHERE` filter
(`ProductQueryService.list_products`) over the `products` table's
metadata columns.

## Four separate flows

There are deliberately **four independent flows** that never run in
the same request:

1. **Offline adapter/classifier training**
   (`scripts/generate_training_data.py` + `scripts/train_color_adapter.py`
   + `scripts/train_category_classifier.py`) — run manually, and only
   if you want color-aware ranking and/or category-aware search
   filtering. Generates a synthetic image/caption dataset, precomputes
   frozen CLIP embeddings for it (shared, cached, between both
   trainers), and trains `ColorAdapter` and/or `CategoryClassifier` on
   those cached vectors. Produces `backend/data/color_adapter.pt`
   and/or `backend/data/category_classifier.pt`. Entirely optional;
   nothing else depends on either having been run.

2. **Offline indexing** (`scripts/build_embeddings.py`) — run manually
   whenever the catalog changes (or after (re)training the color
   adapter). Scans `catalog/`, embeds every product photo once
   (through the adapter too, if a checkpoint exists), and replaces the
   `products` table in Postgres (`TRUNCATE` + bulk insert, via
   `IndexingService`/`app/db.py`). This is the expensive step (loading
   CLIP, running inference per image), but it happens outside the
   request/response cycle.

3. **Online search** (`POST /search`, served by FastAPI) — the CLIP
   model (and color adapter / category classifier, if present) is
   loaded **once at API startup** (see `app/main.py`'s `lifespan`
   handler), which also opens the Postgres connection pool
   (`ProductQueryService`). The catalog itself is *not* loaded at
   startup or cached anywhere in the process -- each request pays for
   encoding *one* uploaded image, optionally classifying it
   (`match_category`/`match_color`), and one pgvector nearest-neighbor
   SQL query (`embedding <=> $query`, via the HNSW index) that does the
   ranking inside Postgres and returns only the top-K rows.

4. **Online metadata browsing** (`GET /products`, served by FastAPI) —
   no CLIP involved at all, just a plain SQL `WHERE` filter
   (`ProductQueryService.list_products`) over the `products` table's
   `category`/`color` columns. The simplest and fastest of the four
   flows.

Keeping the offline flows separate from the online ones is what makes
"every request queries Postgres directly, nothing cached in the app"
viable without a per-request performance hit: the `products` table
(and its HNSW index) is rebuilt in batch by flow 2, so flows 3 and 4
only ever do cheap, index-backed reads. It's also why flow 2 must be
re-run after flow 1 changes the color adapter checkpoint — otherwise
the catalog's stored embeddings and freshly-adapted query embeddings
would be in different (non-comparable) vector spaces. (The category
classifier doesn't have this constraint -- it only classifies the
*uploaded* image at query time, never touches stored catalog
embeddings, so retraining it takes effect immediately on API restart,
no reindex needed.)

## Why embeddings are pre-normalized

`ClipModel.encode_image` L2-normalizes every embedding it produces
(image norm divided by its own L2 norm) before returning it. Once both
vectors in a comparison are unit-length, cosine similarity reduces to
a plain dot product, and pgvector's cosine-distance operator (`<=>`,
used by `ProductQueryService.search_similar`) is defined as
`1 - cosine_similarity` -- which is why the service does
`score = 1 - distance` to turn pgvector's output back into the
similarity score the API returns. This doesn't change the correctness
of using cosine distance (it's mathematically equivalent either way),
but it does mean the stored vectors are ready to compare directly and
consistently, regardless of which layer does the comparison.

`ColorAdapter.forward` preserves this invariant: it adds a learned
residual to the input embedding and then re-normalizes the result, so
adapted embeddings are unit-length too, whether or not a checkpoint is
loaded.

## Layered responsibilities

| Layer | File | Responsibility |
|---|---|---|
| Model | `app/models/clip_model.py` | Owns the HF CLIP model/processor. The *only* place that imports `torch`/`transformers` for inference. Encodes images and text, and does zero-shot classification (`classify`). |
| Model | `app/models/color_adapter.py` | `ColorAdapter` (small residual head) + `load_color_adapter`, which loads a trained checkpoint or returns `None`. |
| Model | `app/models/category_classifier.py` | `CategoryClassifier` (linear head) + `load_category_classifier`, same load-or-`None` pattern. |
| Service | `app/services/embedding_service.py` | Turns raw bytes or a file path into an embedding, applying the color adapter (if loaded) after CLIP. Validates images. |
| Service | `app/services/attribute_classifier_service.py` | Classifies an *uploaded* image's own category/color (zero-shot for color, `CategoryClassifier` for category) -- a different concern from ranking. |
| Service | `app/services/indexing_service.py` | **Write path.** Knows the catalog folder layout and the Postgres `products` table schema. Scans `catalog/`, embeds every image, and replaces the table (run offline, by `scripts/build_embeddings.py`, never by the API). |
| Service | `app/services/product_query_service.py` | **Read path.** Every SQL query `/search` and `/products` need: pgvector nearest-neighbor search (`search_similar`, category/color OR-filtered), plain metadata filtering (`list_products`, AND), and small helper queries (`count`, `distinct_categories`, `distinct_colors`). No caching, no in-memory catalog -- every call hits Postgres. |
| Infra | `app/db.py` | Owns the psycopg connection pool, registers pgvector's Python adapter, and ensures the `vector` extension/`products` table/HNSW index exist. The only module that imports psycopg. |
| Infra | `app/auth.py` | `require_api_key`, a FastAPI dependency checking the `X-API-Key` header against `API_KEY`. Applied at the router level in `api/search.py`/`api/products.py`, not per-route. |
| Infra | `app/rate_limit.py` | The single Redis-backed `slowapi` `Limiter` instance, keyed by client IP. Endpoints import it to set their own `@limiter.limit(...)`; `main.py` wires the shared exception handler/middleware once. |
| API | `app/api/search.py` | HTTP concerns only: validates the upload, calls services, maps errors to HTTP status codes. Router-level auth + a `SEARCH_RATE_LIMIT` limit (CLIP inference is the expensive part). |
| API | `app/api/products.py` | HTTP concerns only: plain metadata filtering, no image/embedding involved at all. Router-level auth + a looser `PRODUCTS_RATE_LIMIT`. |
| Entrypoint | `app/main.py` | Wires everything together once at startup (`lifespan`): opens the DB pool, loads the ML models, registers the rate-limit exception handler/middleware, exposes the FastAPI `app`. Does *not* load the catalog -- there's nothing to load, every request queries Postgres. |
| Script | `scripts/build_embeddings.py` | CLI entrypoint for the offline indexing flow. |
| Script | `scripts/generate_training_data.py` | CLI entrypoint that generates the synthetic color/category training set. |
| Script | `scripts/train_color_adapter.py` | CLI entrypoint for the offline color-adapter-training flow. |
| Script | `scripts/train_category_classifier.py` | CLI entrypoint for the offline category-classifier-training flow. Reuses `train_color_adapter.py`'s manifest/embedding-cache helpers directly rather than duplicating them. |
| Script | `scripts/import_real_photos.py` | Merges real, labeled photos (`backend/data/real_training/<Category>/`) into `training/manifest.json`, so the training scripts above see a mix of synthetic and real examples without any change to their own logic. |

Each layer only talks to the layer directly below it, so, for example,
the storage/ranking approach moved from a JSON file + in-memory
`sklearn` cosine similarity to PostgreSQL/pgvector doing the ranking
itself, by replacing `SimilarityService`/`catalog_filter` with
`ProductQueryService` — the API and model layers, and
`EmbeddingService`, were untouched; `api/search.py` still just hands a
query embedding to one service and gets `SearchResult`s back, with no
idea whether ranking happens in Python or SQL. Likewise,
`EmbeddingService` is the *only* place that knows whether a color
adapter is active — everything downstream of it (the API route,
`ProductQueryService`) just sees embeddings and has no idea whether
they came from raw CLIP or an adapted space. Symmetrically,
`AttributeClassifierService` is the only place that knows *how*
category/color get predicted for an uploaded image (zero-shot vs.
trained classifier) — `api/search.py` just gets back two optional
strings and passes them straight through to
`ProductQueryService.search_similar` as SQL filter arguments.

## Auth and rate limiting

Both are cross-cutting concerns applied at the *router* level, not
per-route, so individual endpoint functions in `api/search.py`/
`api/products.py` stay focused on their own logic:

- **Auth** (`app/auth.py`): `APIRouter(..., dependencies=[Depends(require_api_key)])`
  runs `require_api_key` before every route on that router. It's a
  single shared-secret check (`X-API-Key` header == `API_KEY`) — no
  user accounts, no token issuance/expiry, since every caller is a
  trusted service (this POC has no concept of "logged in as a
  particular user"). Declared via `fastapi.security.APIKeyHeader`
  rather than a manual header read so FastAPI also registers it as a
  proper OpenAPI security scheme (the "Authorize" button in `/docs`).

- **Rate limiting** (`app/rate_limit.py`): one `slowapi.Limiter`,
  keyed by client IP (not by API key -- with one shared key, keying by
  it would bucket every caller together instead of limiting each one).
  It's backed by Redis (`storage_uri=REDIS_URL`) rather than
  in-process memory for the same reason storage moved to Postgres
  earlier: a counter that only lives in one worker's memory stops
  being a real limit the moment there's more than one worker/replica.
  `main.py` registers the shared exception handler
  (`RateLimitExceeded` -> 429) and `SlowAPIMiddleware` once, at app
  creation; each route then declares its own limit --
  `@limiter.limit(SEARCH_RATE_LIMIT)` / `@limiter.limit(PRODUCTS_RATE_LIMIT)`,
  both env-overridable constants in `app/config.py` (defaults
  `"20/minute"`/`"60/minute"`) -- since `/search` (CLIP inference) and
  `/products` (a SQL filter) warrant different limits.

One implementation wrinkle worth noting: `api/search.py` and
`api/products.py` deliberately do **not** use
`from __future__ import annotations` (unlike the rest of this
codebase). With that import active, all type annotations become
strings (PEP 563), and `@limiter.limit(...)`'s wrapping of the
endpoint function breaks FastAPI's ability to resolve
`ForwardRef('UploadFile')` back to the real `UploadFile` class at
import time -- `from __future__ import annotations` and slowapi's
decorator-based API don't compose cleanly through FastAPI's route
introspection. Every other module keeps the future import; only these
two routers omit it.

## Similarity ranking, step by step

`ProductQueryService.search_similar` runs one SQL query per request --
no catalog is ever loaded into Python:

1. The query embedding is bound as a `%s::vector` parameter (an
   explicit cast: pgvector's array→vector cast only applies
   automatically in assignment context, like an `INSERT`, not inside
   an operator expression -- without it Postgres would bind a plain
   Python list as `double precision[]` and `vector <=> double
   precision[]` has no matching operator).
2. If `match_category`/`match_color` predicted a label, it's added as
   a `WHERE category ILIKE %s OR color ILIKE %s` clause (OR -- see
   [Filtering: AND vs OR](#filtering-and-vs-or) below).
3. `ORDER BY embedding <=> %s::vector LIMIT %s` -- pgvector's `<=>`
   operator computes cosine *distance* for each row; Postgres uses the
   `products_embedding_idx` HNSW index (created in `app/db.py`) to
   satisfy this `ORDER BY ... LIMIT` without scanning every row, and
   returns only the top-K.
4. Each row's `distance` is flipped back to a similarity score
   (`1 - distance`, see [Why embeddings are pre-normalized](#why-embeddings-are-pre-normalized))
   and mapped to a `SearchResult`.

At this catalog's scale (17 products) an HNSW index is overkill --
brute-force would be just as fast -- but it means the same code path
also scales to a much larger catalog without changing anything in
`ProductQueryService` or the API layer above it.

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

**Net effect:** ranking is unchanged in shape (still cosine
similarity/distance, computed by pgvector's `<=>` operator, unaware
the adapter exists) — only the vector space the embeddings live in
changes, so color differences contribute more to the resulting score.

## Classifying and filtering an uploaded image

Separate concern from the color adapter above: instead of nudging
*ranking*, `match_category`/`match_color` on `POST /search` (see
`AttributeClassifierService`) ask "what category/color is *in* this
uploaded photo," then exclude everything in the catalog that doesn't
share it.

**Color** uses zero-shot CLIP classification (`ClipModel.classify`):
encode a handful of text prompts like `"a photo of something red
colored"` (one per color actually present in the catalog), and pick
whichever has the highest cosine similarity to the uploaded image's
raw (non-adapter) embedding. This works well out of the box -- no
training needed, and it automatically covers whatever colors the
catalog happens to have.

**Category** needed a different approach. The same zero-shot technique
(prompts like `"a photo of shoes"`) was tried first and turned out
unreliable: on a real test image, similarity scores across all four
categories were within 0.02 of each other -- noise, not signal.
Pretrained CLIP has never seen this catalog's abstract solid-color
placeholder shapes and doesn't recognize them as product categories
the way it would recognize an actual photo. So category classification
uses the *same fix* as the color adapter: freeze CLIP, train a small
head (`CategoryClassifier` -- a plain linear layer, since this is
classification, not embedding-space nudging) on the synthetic training
set (`scripts/train_category_classifier.py`, reusing
`train_color_adapter.py`'s cached embeddings). `AttributeClassifierService`
uses the trained classifier if a checkpoint exists, falling back to
the same unreliable zero-shot approach (with a logged warning)
otherwise.

**A concrete lesson from building this:** a classifier trained on
`generate_training_data.py`'s synthetic images can score 97%+ on its
own held-out validation split while still failing badly on the *real*
catalog it's meant to classify, if the two don't actually look alike.
This happened twice during development: the training generator's hat
was a half-circle (`pieslice`) while every real catalog hat is a full
circle, and its shirt had angled collar notches while every real
catalog t-shirt is a plain rectangle. Both were realistic-enough
renders to *look* like reasonable synthetic data in isolation, but the
validation split was drawn from the same (mismatched) distribution as
training, so it couldn't catch the gap -- only checking against actual
catalog images did. Fixed by aligning `generate_training_data.py`'s
shapes (see `CATEGORY_BOX_SIZE` and the `_draw_*` functions) to match
the catalog's real conventions. Moral: when training data is
synthesized to stand in for real inputs, "does it validate well" and
"does it look right" are both necessary checks, but neither is
sufficient on its own -- validate against the actual deployment inputs
too.

## Real photos: a second, harder domain gap

Aligning the synthetic shapes to the catalog (above) fixed
generalization *within* this project's own synthetic-to-placeholder
world. It did nothing for generalization to *real product photos*,
which is a bigger gap: a real photo (natural lighting, texture,
background clutter, multiple objects) is far outside anything the
purely-synthetic training set ever produced.

Verified concretely: fed a real (fairly messy) photo of a sneaker to
the trained `CategoryClassifier`, it predicted "Accessories" (0.307)
over "Shoes" (0.279) -- wrong, and barely more confident than a coin
flip. Fed the same photo to *untrained* zero-shot CLIP, it correctly
favored "Shoes" (0.277, highest). That's the inverse of the earlier
finding: the classifier's linear decision boundary was fit only to
the synthetic embedding distribution, so a real photo's embedding
lands somewhere that boundary was never calibrated for, while
zero-shot CLIP -- pretrained on millions of real photos -- is
well-calibrated for exactly this input. Neither "trained" nor
"zero-shot" is universally better; each is only reliable on the kind
of image it actually learned from.

`scripts/import_real_photos.py` addresses this the direct way: merge
real, labeled photos into the same manifest the training scripts
already read (`backend/data/real_training/<Category>/*.jpg`, imported
into `training/manifest.json` alongside the synthetic entries), so
`train_category_classifier.py` and `train_color_adapter.py` train on
a mix without any change to the training loop itself -- they don't
know or care whether a sample is synthetic or real.

**What one real example proved, and what it didn't:** importing a
single real photo and retraining flipped that exact photo's prediction
from wrong to right. That confirms the import → retrain → predict
mechanism works, but with `n=1` there's no way to distinguish "the
classifier learned something transferable about real shoes" from "the
classifier memorized this one embedding." A real generalization claim
needs either a held-out real photo the classifier never trained on, or
enough real examples per category that a train/val split over *real*
data is meaningful -- the same reasoning that motivated checking the
synthetic classifier against real catalog images in the first place,
one level up.

## Filtering: AND vs OR

`ProductQueryService` has two query methods that look similar but
build opposite `WHERE` clauses, because `/products` and `/search` want
different things when *both* `category` and `color` are given:

- `list_products` (used by `GET /products`): **AND**
  (`WHERE category ILIKE %s AND color ILIKE %s`, each clause only
  added if that filter was given). Both filters given means "narrow to
  this exact facet combination" -- e.g. `category=Shoes, color=red`
  returns only red shoes. This is the ordinary meaning of stacking
  filters in a faceted browse UI.
- `search_similar` (used by `POST /search`'s `match_category`/
  `match_color`): **OR** (`WHERE category ILIKE %s OR color ILIKE %s`).
  Both given means "widen to anything matching either" -- e.g. a shoe
  query with both checked returns every shoe *plus* every item of the
  query's color, not just red shoes. AND would be actively wrong here:
  since these two checkboxes are about *restricting an image search*,
  not stacking independent facets, AND-ing them would mean neither box
  could ever be used to broaden results along the other axis --
  checking both would always be either equal to or narrower than
  checking one, which isn't what "search near this category or this
  color" should mean.

Both methods use `ILIKE` (case-insensitive, exact -- no `%` wildcards
in the pattern) against the same `category`/`color` columns; a
`NULL` `color` column never matches an `ILIKE` comparison, so a
`color` filter naturally excludes colorless products without any extra
`IS NOT NULL` clause. Only the `AND`/`OR` joining the clauses differs.
