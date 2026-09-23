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
 (optional) match_category/match_color: classify the uploaded image
 itself (AttributeClassifierService) and filter the candidate pool
 (catalog_filter.filter_by_category_or_color) *before* ranking
        │
        ▼
 Cosine similarity vs every embedding in the (possibly filtered) catalog
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
why it exists and how it's trained. `match_category`/`match_color`
filtering is also optional (off unless the query params are set) --
see [Classifying and filtering an uploaded image](#classifying-and-filtering-an-uploaded-image)
below. `GET /products` is a separate, simpler path: it never touches
CLIP at all, just filters `embeddings.json`'s metadata directly.

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
   (through the adapter too, if a checkpoint exists), and writes
   `backend/data/embeddings.json`. This is the expensive step (loading
   CLIP, running inference per image), but it happens outside the
   request/response cycle.

3. **Online search** (`POST /search`, served by FastAPI) — the CLIP
   model (and color adapter / category classifier, if present) is
   loaded **once at API startup** (see `app/main.py`'s `lifespan`
   handler) and kept in memory. Each request pays for encoding *one*
   uploaded image, optionally classifying it (`match_category`/
   `match_color`), and a cosine-similarity matrix multiply against the
   (small, filtered-or-not) catalog — all fast enough for a laptop CPU.

4. **Online metadata browsing** (`GET /products`, served by FastAPI) —
   no CLIP involved at all, just an in-memory filter over
   `embeddings.json`'s `category`/`color` fields. The simplest and
   fastest of the four flows.

Keeping the offline flows separate from the online ones is what makes
the JSON-file approach viable: the "database" (embeddings.json) is
rebuilt in batch, and the API only ever reads it. It's also why flow 2
must be re-run after flow 1 changes the color adapter checkpoint —
otherwise the catalog's stored embeddings and freshly-adapted query
embeddings would be in different (non-comparable) vector spaces. (The
category classifier doesn't have this constraint -- it only classifies
the *uploaded* image at query time, never touches stored catalog
embeddings, so retraining it takes effect immediately on API restart,
no reindex needed.)

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
| Model | `app/models/clip_model.py` | Owns the HF CLIP model/processor. The *only* place that imports `torch`/`transformers` for inference. Encodes images and text, and does zero-shot classification (`classify`). |
| Model | `app/models/color_adapter.py` | `ColorAdapter` (small residual head) + `load_color_adapter`, which loads a trained checkpoint or returns `None`. |
| Model | `app/models/category_classifier.py` | `CategoryClassifier` (linear head) + `load_category_classifier`, same load-or-`None` pattern. |
| Service | `app/services/embedding_service.py` | Turns raw bytes or a file path into an embedding, applying the color adapter (if loaded) after CLIP. Validates images. |
| Service | `app/services/attribute_classifier_service.py` | Classifies an *uploaded* image's own category/color (zero-shot for color, `CategoryClassifier` for category) -- a different concern from ranking. |
| Service | `app/services/catalog_filter.py` | Pure category/color matching over a list of `ProductRecord`. Two functions, AND and OR -- see below. |
| Service | `app/services/indexing_service.py` | Knows the catalog folder layout and the embeddings.json schema. Builds and loads the index. |
| Service | `app/services/similarity_service.py` | Pure ranking logic: cosine similarity + top-K sort. No I/O. |
| API | `app/api/search.py` | HTTP concerns only: validates the upload, calls services, maps errors to HTTP status codes. |
| API | `app/api/products.py` | HTTP concerns only: plain metadata filtering, no image/embedding involved at all. |
| Entrypoint | `app/main.py` | Wires everything together once at startup (`lifespan`), exposes the FastAPI `app`. |
| Script | `scripts/build_embeddings.py` | CLI entrypoint for the offline indexing flow. |
| Script | `scripts/generate_training_data.py` | CLI entrypoint that generates the synthetic color/category training set. |
| Script | `scripts/train_color_adapter.py` | CLI entrypoint for the offline color-adapter-training flow. |
| Script | `scripts/train_category_classifier.py` | CLI entrypoint for the offline category-classifier-training flow. Reuses `train_color_adapter.py`'s manifest/embedding-cache helpers directly rather than duplicating them. |

Each layer only talks to the layer directly below it, so, for example,
swapping the storage format from JSON to PostgreSQL later only
requires changing `IndexingService` — the API and model layers stay
untouched. Likewise, `EmbeddingService` is the *only* place that knows
whether a color adapter is active — everything downstream of it (the
API route, `SimilarityService`) just sees embeddings and has no idea
whether they came from raw CLIP or an adapted space. Symmetrically,
`AttributeClassifierService` is the only place that knows *how*
category/color get predicted for an uploaded image (zero-shot vs.
trained classifier) — `api/search.py` just gets back two optional
strings and hands them to `catalog_filter`.

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

## Filtering: AND vs OR

`catalog_filter.py` has two functions that look similar but implement
opposite combination rules, because `/products` and `/search` want
different things when *both* `category` and `color` are given:

- `filter_by_category_and_color` (used by `GET /products`): **AND**.
  Both filters given means "narrow to this exact facet combination" --
  e.g. `category=Shoes, color=red` returns only red shoes. This is the
  ordinary meaning of stacking filters in a faceted browse UI.
- `filter_by_category_or_color` (used by `POST /search`'s
  `match_category`/`match_color`): **OR**. Both given means "widen to
  anything matching either" -- e.g. a shoe query with both checked
  returns every shoe *plus* every item of the query's color, not just
  red shoes. AND would be actively wrong here: since these two
  checkboxes are about *restricting an image search*, not stacking
  independent facets, AND-ing them would mean neither box could ever
  be used to broaden results along the other axis -- checking both
  would always be either equal to or narrower than checking one, which
  isn't what "search near this category or this color" should mean.

Both functions share the same underlying match predicates
(`_category_matches`/`_color_matches`: case-insensitive, exact,
`color=None` never matches a product with no color set) -- only the
`and`/`or` at the end differs.
