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

## Six separate flows

There are deliberately **six independent flows** that never run in
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
   `category`/`color` columns. The simplest and fastest of the five
   flows.

5. **Online product creation** (`POST /products`, served by FastAPI) —
   embeds the uploaded photo (same `EmbeddingService` transformation as
   flow 3, so the new product is immediately comparable against
   everything flow 2 already indexed) and calls
   `IndexingService.add_product`, which writes `catalog/<sku>/` (image
   + metadata.json) *and* inserts one row into `products` -- both, not
   just the DB row, specifically so flow 2 doesn't silently delete this
   product on its next full rebuild (see
   [Why product creation also writes to disk](#why-product-creation-also-writes-to-disk)
   below). Synchronous: the response only returns once the product is
   fully stored.

6. **Bulk product import** (`POST /products/import`, served by
   FastAPI, but *processed* by a separate `worker` process) —
   structurally validates a whole batch (JSON shape, field-level
   constraints, file types) synchronously, then publishes one message
   per product onto a durable RabbitMQ queue and returns immediately,
   without waiting for any embedding to happen. A `worker` container
   (`scripts/run_worker.py` + `app/jobs.py`) consumes that queue,
   calling the exact same `IndexingService.add_product` flow 5 calls
   inline. See [Bulk product import](#bulk-product-import) below for
   why this one flow splits across two processes (and two brokers --
   RabbitMQ here, not the Redis flow 3's rate limiting uses) instead of
   running entirely within the request like flows 3-5.

Keeping flows 1-4 separate from each other is what makes "every read
request queries Postgres directly, nothing cached in the app" viable
without a per-request performance hit: the `products` table (and its
HNSW index) is rebuilt in batch by flow 2, so flows 3 and 4 only ever
do cheap, index-backed reads. It's also why flow 2 must be re-run
after flow 1 changes the color adapter checkpoint — otherwise the
catalog's stored embeddings and freshly-adapted query embeddings would
be in different (non-comparable) vector spaces. (The category
classifier doesn't have this constraint -- it only classifies the
*uploaded* image at query time, never touches stored catalog
embeddings, so retraining it takes effect immediately on API restart,
no reindex needed.) Flows 5 and 6 are the odd ones out precisely
because they're writes that happen online -- see the dedicated
sections below for how each stays consistent with flow 2 rather than
fighting it, and how 6 additionally avoids blocking on the ML work 5
does inline.

## Why product creation also writes to disk

`IndexingService.add_product` (flow 5) writes `catalog/<sku>/image.{jpg,png}`
and `catalog/<sku>/metadata.json` -- the exact same layout every other
product already has -- *in addition to* inserting the new row into
Postgres. That's not redundancy for its own sake; it's what keeps flow
5 from being quietly undone by flow 2.

`IndexingService.build_index` (flow 2, `scripts/build_embeddings.py`)
doesn't merge or upsert -- it scans `catalog/` and **replaces the
entire `products` table** (`TRUNCATE` + bulk insert) from whatever it
finds there. That's a deliberate, pre-existing design choice: it makes
"the catalog folder is the single source of truth for a full rebuild"
unambiguous, with no drift between what's on disk and what's in the
table. But it means anything that only ever reached the database --
skipping the catalog folder -- would vanish the next time someone runs
a routine reindex, with no error or warning, because from flow 2's
point of view that product never existed.

So `add_product` treats the catalog folder as no less authoritative
than the database: both get written, in that order (folder first,
since it's simpler to detect "already exists" and to clean up on
failure), and if anything after the folder write fails -- writing
`metadata.json`, embedding the photo, or the DB insert itself -- the
partially-created folder is removed again (`shutil.rmtree`) before the
error propagates. A caller never sees `201 Created` for a product that
isn't durably in *both* places, and a failed request never leaves an
orphaned folder for a later `build_index` run to trip over.

The tradeoff this accepts: `add_product` isn't atomic across the
filesystem and Postgres in the strict sense (a crash between the folder
write and the DB insert -- as opposed to a caught exception, which
*is* cleaned up -- could theoretically leave an orphaned folder). For
a POC this is an acceptable gap; a production version would want
either a two-phase write (temp folder + atomic rename) or to treat
`catalog/` as a cache rebuildable from Postgres instead of the other
way around.

## Bulk product import

`POST /products/import` (flow 6) exists for one reason: `POST /products`
(flow 5) does real work inline -- CLIP inference plus a filesystem
write -- and that's fine for one product per request, but doesn't
scale to "import a few hundred products from a Magento export" without
either a very long-lived HTTP request or the client looping the
single-product endpoint hundreds of times, each paying full HTTP
overhead and leaving no good way to know which items landed if one
partway through fails.

**Split across two processes, not two code paths.** The actual
per-product work -- embed the photo, write `catalog/<sku>/`, insert
the DB row -- is `IndexingService.add_product`, the *same* method flow
5 calls. Bulk import doesn't reimplement or wrap it differently; it
just calls it from a different process, later:

- **Producer** (`app/queue.py`, imported by `api/products.py`, so part
  of the API process): opens a short-lived `pika` connection per call
  and publishes one JSON message per product (the image bytes
  base64-encoded inside it -- AMQP messages are just bytes) to a
  *durable* queue, `delivery_mode=2` (persistent). The API process
  never imports `app.jobs`/`IndexingService`/`ClipModel` for this
  feature at all, keeping the "API process never touches CLIP for
  anything but `/search`" property flows 3-5 already had -- there's no
  string-reference trick needed here (unlike an in-process task queue)
  since the message itself, not a Python callable, is what crosses the
  process boundary.
- **Consumer** (`app/jobs.py` + `scripts/run_worker.py`, the `worker`
  docker-compose service -- never imported by the API process):
  `run_worker.py` calls `jobs.init_services()` once at process
  startup, loading CLIP and opening a DB pool exactly like
  `app.main`'s `lifespan` does for the API -- then opens its own
  `pika` connection and consumes from the same queue, decoding each
  message (`jobs.handle_message`) and calling
  `jobs.import_product_job`, which delegates to
  `IndexingService.add_product`.

**Why RabbitMQ, not the Redis already used for rate limiting.** Two
different jobs with two different durability requirements, kept in two
different failure domains on purpose. Rate-limit counters
(`app/rate_limit.py`) are ephemeral by design -- losing them on a
restart just means limits reset to zero, a non-event. A queued product
import is real, expected work: a caller (a Magento export) published
it and expects it to actually happen, eventually, even across a worker
restart. Redis *can* be made to serve as a durable queue (`rpoplpush`
patterns, or a library like RQ built on top of it), but that's Redis
doing something outside its core design center; RabbitMQ's whole job
is exactly this -- durable queues, per-message acknowledgment, and
automatic redelivery of unacked messages -- built in, not layered on.
Coupling the queue to the rate-limiting Redis would also mean "Redis is
down" and "the import pipeline is down" become the same incident,
which they have no real reason to be.

**One message in flight at a time, acked only after success.**
`scripts/run_worker.py` sets `channel.basic_qos(prefetch_count=1)` and
only calls `channel.basic_ack(...)` *after* `jobs.handle_message`
returns without raising. Concurrency was never the goal -- this
worker holds a single loaded CLIP model and a single psycopg
connection pool, initialized once at startup specifically so jobs
don't pay that cost repeatedly, and nothing about either is safe for
concurrent use, so `prefetch_count=1` isn't a throughput compromise,
it's the only correct setting. The ack-after-success ordering is what
actually buys something: if `import_product_job` hits a genuinely
unexpected error (Postgres unreachable, a bug), it re-raises instead
of swallowing it (unlike the two *expected* failure modes below); that
exception propagates out of `_on_message`, out of `start_consuming()`,
and out of the script entirely -- the message is never acked, so
RabbitMQ holds onto it and redelivers it once a worker reconnects, and
`docker-compose.yml`'s `restart: unless-stopped` on the `worker`
service is what makes "a worker reconnects" actually happen
automatically rather than requiring a human to notice and restart it.
Verified directly: stopped the `worker` container, queued a product
via `POST /products/import`, confirmed the RabbitMQ management API
showed `messages_persistent: 1` on the `product_import` queue while
the worker was down, then restarted `worker` and confirmed the message
was consumed and the product imported within about a second of
reconnecting.

**Validation is split deliberately between the two sides.** Everything
checkable without touching Postgres or CLIP -- is the JSON well-formed,
do `products`/`files` counts match, does each item satisfy
`BulkProductItem`'s constraints, is the batch within
`MAX_BULK_IMPORT_ITEMS`, is each file a supported content type --
happens synchronously in `api/products.py`, *before* anything is
published, so an obviously malformed batch fails the whole request
with a specific error instead of partially queuing garbage. Everything
that genuinely requires the DB or CLIP -- does this `sku` already
exist, is this file actually a decodable image -- can only be
discovered once a worker processes the job, so it happens there
instead, and (being a permanent, not transient, failure) is *logged
and acked*, not retried (see below).

**Fire-and-forget is a deliberate scope decision for *outcomes*, not
for *durability*, and not an oversight.** There is no
`GET /products/import/{batch_id}` status endpoint. Each job sets
`app.logging_config.request_id_var` to
`f"import-{batch_id}-{item_index}"` before calling `add_product` (the
same mechanism `RequestContextMiddleware` uses for HTTP requests, just
driven manually here since there's no request), so every log line for
one item -- success, or a caught `FileExistsError`/`ValueError` that
gets acked as "done" because retrying it would just fail identically
-- is correlated and grep-able (`docker-compose logs worker | grep
<batch_id>`), but that correlation lives only in logs, not in any
queryable job-status store. What bulk import does *not* fire-and-forget
is whether the work happens at all -- that's the ack/redelivery
mechanism above. A production version handling a partner integration
like this would likely still want a persisted per-item status (a
result table, or a dead-letter queue for permanent failures instead of
just logging them) so Magento itself could ask "did SKU X import
successfully" instead of a human grepping logs. Revisit if the answer
to "does Magento need to know per-item outcomes" changes.

## Magento connector

`magento/app/code/VisualSearch/Connector/` is a client of this API; it
adds no server-side code here. Each Magento feature maps onto an
existing endpoint:

| Magento feature | Endpoint | Why |
|---|---|---|
| Manual "Sync to Visual Search" button; on-save sync ("immediate" mode) | `POST /products` | Synchronous, so the outcome (`201` created / `409` exists, followed by `PUT /products/{sku}` / `400` bad input) is known immediately and logged |
| Product-grid mass action; on-save sync ("queue" mode) | `POST /products/import` | Async batches (≤100). Magento publishes to its own DB queue and a consumer sends the batches, so an admin click never waits on CLIP |
| Storefront "Search by Image" | `POST /search` | Result SKUs are mapped back to Magento products and re-checked against the storefront (enabled, visible, in-store) |
| Chat widget ("Shopping assistant") | `POST /agent/chat` | Browser -> Magento JSON endpoint -> API, so the API key and shopper id stay server-side; the SKUs in the reply are mapped back to storefront products like search results |
| Shopper tracking (view / add to cart / purchase) | `POST /events` | Queued in Magento and forwarded by a consumer, so the API can never slow a page or checkout |
| "Recommended for you" block | `GET /recommendations` | Loaded by AJAX per visitor -- the host page is full-page-cached and shared, so personal results can't be rendered into it |

Places where the API's contract (and Magento's page cache) shape the module:

- **SKU mapping.** The API only accepts lowercase-slug SKUs
  (`SKU_PATTERN`), Magento SKUs are free-form. The module normalizes
  deterministically (`Shoe_Red 42` → `shoe-red-42`) and stores the pair
  in `visual_search_product_map`, because `/search` returns *API* SKUs.
  Two Magento SKUs that normalize to the same slug collide (the second
  overwrites the first via the update path).
- **`queued` is a terminal status.** `POST /products/import` is
  fire-and-forget (see [Bulk product import](#bulk-product-import)):
  Magento can record "accepted, batch `<id>`" but never the per-product
  outcome, which lives in the worker's logs. The sync-log grid shows the
  `batch_id` to grep for. This is the "does Magento need per-item
  outcomes" question that section leaves open; answering yes would mean
  a status endpoint on this API.
- **Update, but no delete.** `PUT /products/{sku}` replaces a product's
  metadata and (optionally) its photo. `SyncService::syncNow` tries
  `POST /products` first and, on `409`, follows up with the `PUT`, so a
  re-save of an already-synced product refreshes the index (logged
  `success`). The bulk worker (`app/jobs.py`) does the same when
  `add_product` raises `FileExistsError`. There is no `DELETE`: deleted
  Magento products stay in the index (the storefront filter hides them).

- **Views come from the browser; carts and orders from PHP.** With
  full-page cache on, a product page is usually served without reaching
  PHP, so a server-side "product viewed" observer would miss most views.
  A small JS beacon on the product page posts them instead
  (`js/track-view.js` -> `visualsearch/event/track`, once per product per
  session). Add-to-cart and order placement are uncached flows, so plain
  observers handle them.
- **Shopper identity.** Logged-in customers are `c<customer id>`; guests get
  `g<random>` in a first-party `vs_shopper` cookie. A guest who logs in gets
  a new id, so their guest history doesn't carry over (they see the popular
  fallback again). Cookie use should be covered by your consent policy.

Operational note: rate limits are per client IP, and the Magento server
is the only client the API sees, so *all shoppers share one bucket*
(`SEARCH_RATE_LIMIT`, default `20/minute`). Raise it (or key the limiter
on a forwarded client IP) before real traffic. Bulk batches wait out a
`429` using `Retry-After`, up to twice.

The module was verified against a mock server that reproduces this API's
route signatures; it has not yet been smoke-tested inside a running
Magento install (see `magento/README.md`).

## Testing

`backend/tests/` is a pytest unit suite that needs no running
infrastructure: Postgres, CLIP, RabbitMQ, and the service layer are
replaced by hand-written fakes (`tests/fakes.py`), so the tests assert
this app's own logic -- the SQL each query builds (OR for search vs AND
for list filters, bound parameters, distance → similarity), cleanup of a
half-written product folder, the queue message format and persistence,
the worker's swallow-vs-redeliver failure split, request-ID
correlation, and each route's validation, auth, and rate limiting.

API tests go through the real `app.main.app` (real routers, middleware,
and limiter with in-memory storage) with fakes placed on `app.state`
where `lifespan` would put the real services, so wiring and middleware
order are exercised too. `TestClient` is deliberately not used as a
context manager, which would run `lifespan` (CLIP load + Postgres).

Not covered by design: `ClipModel`, `db.create_pool`, the worker
consumer loop, and the training scripts. They need the real model or a
pgvector Postgres, so they belong in integration tests (a Postgres
container is the natural next step).

## Personalized recommendations

`POST /events` records what a shopper does; `GET /recommendations` turns
their recent history into product suggestions. It reuses the CLIP
embeddings already in the `products` table -- there is no model to train.

**Algorithm** (`services/recommendation_service.py`):

1. Load the shopper's newest events (at most `RECOMMEND_MAX_HISTORY`,
   within `RECOMMEND_WINDOW_DAYS`) that point at *indexed* products.
2. Give each event a weight: `EVENT_WEIGHTS[type] * 0.5 ** (age / half_life)`
   -- a purchase (5) outweighs an add-to-cart (3) outweighs a view (1), and
   an event loses half its pull every `RECOMMEND_HALF_LIFE_DAYS` (14).
3. Sum `weight * embedding` and L2-normalize: the **taste vector**. Because
   every embedding is a unit vector in one shared space (see
   [Why embeddings are pre-normalized](#why-embeddings-are-pre-normalized)),
   this is the weighted "centre" of what the shopper looks at, and cosine
   similarity to it is a sensible relevance score.
4. Ask pgvector for the nearest products (`ORDER BY embedding <=> taste`),
   excluding anything the shopper **purchased or carted** (plus caller-given
   `exclude_sku`s). Merely *viewed* products stay eligible.

**Cold start.** A shopper with no usable history gets `strategy: "popular"`:
products ranked by weighted engagement across *all* shoppers. If no events
exist anywhere yet, `strategy: "none"` and an empty list. The strategy is
in the response so a UI can label or hide the block accordingly.

**Design choices worth knowing:**

- *One centroid, not clusters.* A shopper who likes both shoes and bags gets
  a vector between the two, which can land on neither. Fine for one dominant
  interest; the natural upgrade is several centroids (cluster the history,
  query each, interleave) or MMR re-ranking for diversity.
- *No foreign key* from `shopper_events` to `products`: `build_index`
  TRUNCATEs `products` on every reindex, and an event may precede its
  product's indexing. Events for unknown SKUs are stored and ignored until
  the product exists.
- *Anonymous ids only.* `shopper_id` is a client-chosen opaque token; the
  service never sees who anyone is. `DELETE /shoppers/{id}/events` erases a
  shopper. Events older than the window are ignored but not yet purged --
  add a retention job before storing real traffic long-term.
- *Per-IP rate limits are high* (`EVENTS_RATE_LIMIT`, `RECOMMENDATIONS_RATE_LIMIT`)
  because a storefront sends every shopper's traffic from one IP.
- *Popularity is unpersonalized and unweighted by recency* -- it is a
  fallback, not a ranking model.

**Verification.** Besides the unit tests, the SQL was run against a real
pgvector Postgres (decay ordering, exclusions, cold start, erasure), and
the Magento client was exercised against the real routes.

## Local AI agent

`services/agent_service.py` adds natural-language chat on top of the
existing services. It is a thin orchestrator: retrieval stays with
CLIP and pgvector, and a local LLM (Ollama, default `qwen3:8b`) only
decides which tool to call and how to phrase the answer. The
step-by-step plan is in `LOCAL_AGENT_INTEGRATION.txt`; this covers the
loop and its endpoint.

```
messages = [system, *history, user]
repeat up to AGENT_MAX_STEPS:
    reply = ollama /api/chat (messages, TOOL_SCHEMAS)
    no tool_calls  -> return reply (final answer)
    each tool call -> run_tool -> append {"role": "tool", ...}
cap hit -> "Sorry, I couldn't finish that."
```

Everything the model emits is untrusted, so:

- **Whitelist.** Only the tools in `agent_tools.py` (`search_products`,
  `search_similar_to_image`, `get_recommendations`, `get_product`) can
  run; an unknown name comes back as an `{"error": ...}` the model can read.
- **Validation.** Each tool validates its arguments with pydantic
  (`extra="forbid"`) before touching a service.
- **Errors don't abort.** Tool exceptions are logged and returned to the
  model as `{"error": ...}` so it can retry or answer without the tool.
- **Read-only tools.** No create/update/delete is exposed, which limits
  what prompt injection via product text can do.
- **Shopper isolation.** `get_recommendations` uses the `shopper_id`
  passed to `AgentService.chat()` from the server-side session, never the
  model's argument.
- **Images stay server-side.** Uploads go in an `ImageStore`; the model
  only sees an opaque ref like `img_1`.
- **Non-blocking.** The LLM call uses an async `httpx` client and the
  sync tools run via `run_in_threadpool`, so the event loop is never blocked.

### `POST /agent/chat`

Multipart form: `message` (required), `file` (optional image),
`session_id` and `shopper_id` (optional). Requires `X-API-Key`, is
rate-limited by `AGENT_RATE_LIMIT`, and reuses `read_validated_image`
for uploads. Response:

```
{"reply": "...", "products": [{sku, name, price, category, color?, score?}],
 "session_id": "...", "steps": [...]}   # steps only if AGENT_INCLUDE_TRACE
```

- `products` is taken from the most recent tool call that returned
  products, so a UI renders cards from real data rather than model prose.
- An uploaded image is stored server-side; the model is told only
  `image_ref=img_N`.
- `session_id` (generated if omitted) keys the conversation history in
  Redis (`agent_sessions.py`): plain user/assistant text only, last
  `AGENT_HISTORY_MAX_MESSAGES` (10), expiring after
  `AGENT_SESSION_TTL_SECONDS` of inactivity. If Redis is down the chat
  still works, just without memory.
- `shopper_id` comes from the (API-key-holding) storefront and is
  injected into `get_recommendations` server-side.
- Swagger (`/docs`): the route appears under the `agent` tag; use
  **Authorize** for the API key, then **Try it out** (omit the optional
  fields to leave them out). Needs Ollama, Postgres and Redis running.
- Ollama runs natively on the host. In Docker, `docker-compose.yml`
  sets `OLLAMA_URL=http://host.docker.internal:11434` (inside the
  container `localhost` is the container itself); a local `uvicorn`
  uses the default `http://localhost:11434`. "LLM backend error: All
  connection attempts failed" in the logs means this URL is wrong or
  Ollama isn't running (`ollama serve`).
- Ollama unreachable/erroring -> 503; unexpected errors -> 500 with a
  generic message.

Settings (`config.py`): `OLLAMA_URL`, `AGENT_MODEL`, `AGENT_MAX_STEPS`,
`AGENT_RATE_LIMIT`, `AGENT_SESSION_TTL_SECONDS`,
`AGENT_HISTORY_MAX_MESSAGES`, `AGENT_MESSAGE_MAX_LENGTH`,
`AGENT_INCLUDE_TRACE`. Tests use a scripted fake LLM client and fake
Redis, so no Ollama is needed.

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
| Service | `app/services/indexing_service.py` | **Write path.** Knows the catalog folder layout and the Postgres `products` table schema. `build_index`: offline, whole-catalog replace (run by `scripts/build_embeddings.py`, never by the API). `add_product`: online, single-product add (`POST /products`) -- writes `catalog/<sku>/` *and* inserts one DB row, with rollback (folder removal) on failure. |
| Service | `app/services/product_query_service.py` | **Read path.** Every SQL query `/search` and `/products` need: pgvector nearest-neighbor search (`search_similar`, category/color OR-filtered), plain metadata filtering (`list_products`, AND), and small helper queries (`count`, `distinct_categories`, `distinct_colors`). No caching, no in-memory catalog -- every call hits Postgres. |
| Infra | `app/db.py` | Owns the psycopg connection pool, registers pgvector's Python adapter, and ensures the `vector` extension/`products` table/HNSW index exist. The only module that imports psycopg. |
| Infra | `app/auth.py` | `require_api_key`, a FastAPI dependency checking the `X-API-Key` header against `API_KEY`. Applied at the router level in `api/search.py`/`api/products.py`, not per-route. |
| Infra | `app/rate_limit.py` | The single Redis-backed `slowapi` `Limiter` instance, keyed by client IP. Endpoints import it to set their own `@limiter.limit(...)`; `main.py` wires the shared exception handler/middleware once. |
| Infra | `app/logging_config.py` | Configures the root logger once (`configure_logging`, called first thing in `main.py` *and* `scripts/run_worker.py`); owns `request_id_var` (a `ContextVar`) and the filter that stamps it onto every log record. Pins noisy third-party loggers to `WARNING`. |
| Infra | `app/middleware.py` | `RequestContextMiddleware` -- sets `request_id_var` for HTTP requests. Assigns/propagates a request ID, logs one access-log line per request with timing, sets the `X-Request-ID` response header. Registered as the outermost middleware. (`app/jobs.py` sets the same `ContextVar` its own way, for queued jobs instead of requests.) |
| Infra | `app/queue.py` | **Bulk-import producer**, imported only by `api/products.py`. `enqueue_import_job` opens a short-lived `pika` connection to RabbitMQ (a broker deliberately separate from the rate-limiting Redis) and publishes one durable, persistent JSON message per product -- the API process never imports `app.jobs`/`IndexingService`/`ClipModel` for this feature. |
| Infra | `app/jobs.py` | **Bulk-import consumer**, imported only by `scripts/run_worker.py` (never the API process). `init_services()` loads CLIP + opens a DB pool once, eagerly, for the worker's whole lifetime. `handle_message` decodes one RabbitMQ message body; `import_product_job` calls `IndexingService.add_product`, catching/logging *permanent* failures (`FileExistsError`/`ValueError`, no point redelivering) but re-raising anything unexpected so the message stays unacked and RabbitMQ redelivers it. |
| API | `app/api/search.py` | HTTP concerns only: validates the upload, calls services, maps errors to HTTP status codes. Router-level auth + a `SEARCH_RATE_LIMIT` limit (CLIP inference is the expensive part). |
| API | `app/api/products.py` | HTTP concerns only. `GET /products`: plain metadata filtering, `PRODUCTS_RATE_LIMIT`. `POST /products`: validates the upload + form fields (`SKU_PATTERN`, content type), calls `IndexingService.add_product` synchronously, maps `FileExistsError`/`ValueError` to 409/400, `CREATE_PRODUCT_RATE_LIMIT`. `PUT /products/{sku}`: updates metadata and optionally the photo via `IndexingService.update_product`, maps `FileNotFoundError`/`ValueError` to 404/400. `POST /products/import`: validates a whole batch's *shape* synchronously (JSON, counts, per-item fields, file types), then calls `app.queue.enqueue_import_job` per item and returns `202` without waiting, `BULK_IMPORT_RATE_LIMIT`. All four share router-level auth. |
| Entrypoint | `app/main.py` | Configures logging first, then wires everything together once at startup (`lifespan`): opens the DB pool, loads the ML models, registers the rate-limit and request-logging middleware, exposes the FastAPI `app`. Does *not* load the catalog -- there's nothing to load, every request queries Postgres. Never imports `app.jobs` (see `app/queue.py`'s row above). |
| Script | `scripts/build_embeddings.py` | CLI entrypoint for the offline indexing flow (flow 2). |
| Script | `scripts/run_worker.py` | CLI entrypoint for the `worker` service (flow 6's consumer). Calls `app.jobs.init_services()` once, then consumes from RabbitMQ one message at a time (`prefetch_count=1`), acking only after a successful import -- see [Bulk product import](#bulk-product-import) for why. |
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

## Input validation

Untrusted input is checked at the HTTP edge, before any expensive work:

- **Uploads** (`app/validation.py::read_validated_image`, used by all
  three image routes): declared type allowed -> read in 64 KB chunks
  against `MAX_UPLOAD_BYTES` (413; never trusts `Content-Length`) ->
  non-empty -> Pillow sniffs the real format from the bytes, which must
  be allowed and match the declared type -> pixel count under
  `MAX_IMAGE_PIXELS` (decompression bomb) -> `Image.verify()`. The
  stored filename in `POST /products` comes from the sniffed format.
  `EmbeddingService` also maps `OSError` (truncated data) to the same
  `ValueError` -> 400 as an unidentifiable image.
- **Request size** (`BodySizeLimitMiddleware`): a pure-ASGI check of the
  declared `Content-Length` so oversized bodies are refused before the
  multipart parser spools them. Chunked bodies without the header fall
  back to the per-file cap.
- **Metadata**: `NAME/CATEGORY/COLOR_MAX_LENGTH`, `TEXT_PATTERN` (no
  leading whitespace, no control characters -- these values reach
  `metadata.json`, SQL rows and log lines) and `MAX_PRICE` /
  `allow_inf_nan=False`, shared by the `POST /products` form and
  `BulkProductItem`.

## Logging and request correlation

A third cross-cutting concern, alongside auth and rate limiting, also
applied once rather than woven through every endpoint:

- **`app/logging_config.py`** owns the root logger's setup
  (`configure_logging()`, called once from `main.py` before anything
  else runs -- every module's `logging.getLogger(__name__)` inherits
  this rather than configuring its own handlers). It defines
  `request_id_var`, a `contextvars.ContextVar[str]`, and a
  `logging.Filter` that stamps every emitted `LogRecord` with whatever
  value is currently in that var. It also pins known-noisy third-party
  loggers (`multipart`, `urllib3`, `PIL`, `httpcore`/`httpx`) to
  `WARNING` regardless of the app's own `LOG_LEVEL`, since their
  protocol-level internals (every multipart chunk, every connection-pool
  event) are noise, not debugging signal, at DEBUG.

- **`app/middleware.py`**'s `RequestContextMiddleware` is the *only*
  thing that ever sets `request_id_var` -- once per request, from an
  inbound `X-Request-ID` header if the caller sent one (so a single ID
  can thread through this service's logs *and* an upstream caller's
  own tracing) or a fresh one otherwise. It's registered as the
  **outermost** middleware (`app.add_middleware(...)` called *after*
  `SlowAPIMiddleware` in `main.py` -- Starlette wraps in reverse
  registration order, so the last-added middleware runs first on the
  way in and last on the way out): the request ID must exist before
  rate limiting even runs, and the access-log line needs to see the
  real final status code, including a 401 from `require_api_key` or a
  429 from the rate limiter, not just whatever the inner layers saw.

Because `request_id_var` is a `ContextVar` rather than a value passed
explicitly, every service function downstream of the middleware --
`EmbeddingService`, `AttributeClassifierService`,
`ProductQueryService` -- gets request correlation for free just by
calling `logger.debug(...)`/`logger.info(...)` normally. None of them
know a request ID exists; `docker-compose logs backend | grep
<request-id>` still pulls out every line for one request across all of
them, because the middleware set the context before any of that code
ran and Python's `contextvars` propagate down through async calls
automatically.

`LOG_LEVEL=DEBUG` additionally turns on per-step timing logs in the
three services actually worth timing: `EmbeddingService` (CLIP
inference), `AttributeClassifierService` (the `match_category`/
`match_color` classification), and `ProductQueryService.search_similar`
(the pgvector query). Each logs its own elapsed time, so a slow
`/search` request is diagnosable straight from the logs -- e.g. real
output from a `match_category=true&match_color=true` request shows
attribute classification (686ms) dominating over CLIP embedding
(136ms) and the pgvector query (4ms), not the vector search itself.

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
