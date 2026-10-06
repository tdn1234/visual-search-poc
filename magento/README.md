# VisualSearch_Connector (Magento 2 module)

Connects a Magento 2.4.x store (PHP 8.1+) to the visual-search-poc API in
`../backend`: product sync (single, on save, bulk via queue), a sync-log
admin grid, and search by image on the storefront.

Module code lives in `app/code/VisualSearch/Connector/` -- copy or symlink
`app/code/VisualSearch` into your Magento root's `app/code/`.

## Install

```bash
bin/magento module:enable VisualSearch_Connector
bin/magento setup:upgrade          # creates visual_search_sync_log + visual_search_product_map
bin/magento setup:di:compile       # production mode only
bin/magento cache:flush
```

## Configure

Stores > Configuration > Services > **Visual Search**

| Setting | Notes |
|---|---|
| Enable Module | Master switch: off = no sync, no storefront search UI |
| API Base URL / API Key | The service URL and its `X-API-Key` (key is stored encrypted) |
| Sync Product on Save + mode | `Immediate` calls the API during the save; `Via Magento queue` defers it |
| Bulk Sync Batch Size | 1-100 products per `POST /products/import` (API cap is 100) |
| Color Attribute Code | Attribute sent as the product color (default `color`) |
| Only Match Detected Category / Color | Passes `match_category` / `match_color` to `POST /search` |
| Track Shopper Events | Report views / add-to-carts / purchases (default on once the module is enabled) |
| Show "Recommended for you" + Products to Show | Personalized block on the home and product pages (default off) |
| Show Chat Widget + Assistant Timeout | AI shopping-assistant chat box on every storefront page (default off; needs the service's Ollama agent) |

## Features

| Requirement | Where |
|---|---|
| Admin config | Config section above |
| Sync one product manually | **Sync to Visual Search** button on the product edit page |
| Auto-sync on save | `Observer/ProductSaveObserver` (`catalog_product_save_commit_after`) |
| Bulk sync via Magento queue | Product grid > select products > Actions > **Sync to Visual Search** |
| Sync status logging | Every attempt is a row in `visual_search_sync_log` |
| Log history page | Catalog > **Visual Search Sync Log** (filterable by status/type/SKU/date) |
| Search by image | "Search by Image" header link, plus an upload form on the search results page |
| Shopping assistant chat | Floating chat box (bottom-right) posting to `POST /agent/chat` via `visualsearch/chat/send`; text or a photo, replies with product cards |
| Personalized recommendations | "Recommended for you" block on the home page and product pages (`GET /recommendations`), fed by shopper tracking (`POST /events`) |

### Personalized recommendations

1. Enable **Track Shopper Events** first so history builds up, and run the
   event consumer: `bin/magento queue:consumers:start visualsearch.event.track`
   (or let cron's `consumers_runner` do it).
2. Then enable **Show "Recommended for you"**.

What is tracked: product page **views** (JS beacon, because pages are
full-page-cached), **add to cart**, and **purchases** (order placement).
The shopper is `c<customer id>` when logged in, or a random id in a
first-party `vs_shopper` cookie for guests — no personal data is sent.
The block is filled by AJAX (`visualsearch/recommendation/get`, never
cached), hides itself when there is nothing to show, and on a product page
leaves that product out. New shoppers see the store's popular items.

Events are best-effort: if the API is down they are logged and dropped.
Make sure tracking is covered by your cookie-consent/privacy policy.

### Shopping assistant chat

Enable **Show Chat Widget** (Services > Visual Search > Shopping Assistant).
The widget (`js/chat.js`) posts the message and optional photo to the
module's JSON endpoint `visualsearch/chat/send`, which calls the service's
`POST /agent/chat`. Notes:

- The shopper id is the same `c<id>` / `vs_shopper` cookie id used for
  recommendations, resolved on the server (never sent by the browser), so
  "what would I like?" uses that shopper's history.
- The conversation's `session_id` is kept in `sessionStorage`; the service
  holds the history briefly in Redis.
- Product cards are rebuilt from the service's SKUs and re-checked against
  the storefront (enabled, visible, in-store) -- the same as search. The
  assistant's *text* is the model's, so it may still name a product that
  has no card.
- The upload is validated by content (JPEG/PNG/WEBP, 5 MB). The CSRF
  `form_key` is sent automatically. Responses are never cached.
- A local LLM is slow, hence the separate **Assistant Timeout** (default
  120 s). The service must be able to reach Ollama (see the root README).

### Bulk sync needs a queue consumer

Bulk (and on-save "queue" mode) publishes to the DB-backed queue
`visualsearch.product.sync`. Something must consume it:

```bash
bin/magento queue:consumers:start visualsearch.product.sync
```

or rely on Magento cron's `consumers_runner` (default: runs all consumers).

## Sync log statuses

| Status | Meaning |
|---|---|
| `pending` | Waiting in the Magento queue |
| `queued` | Accepted by the API's async import (`202`). **Final** state for bulk: the API has no status endpoint, so per-product results are only in the service's worker log (grep the `batch_id` shown in the grid) |
| `success` | Created (`201`) or, if it was already indexed, updated (`200`) — single sync |
| `skipped` | Already in the index (`409`) — no longer produced by single sync, which now updates instead |
| `failed` | Product couldn't be built (no price / no JPEG-PNG image), API error, or API unreachable -- see Message |

## Things to know

- **Re-saving an already-synced product updates it.** The single sync tries
  `POST /products`; on `409` it sends `PUT /products/{sku}` with the current
  name, price, category, color and image (log: `success`, "metadata and image
  updated"). Bulk imports are updated by the API's worker the same way (see
  its log). The API has no delete endpoint: products deleted in Magento stay in the index; storefront results are
  re-checked against Magento, so they're simply not shown.
- **SKUs are normalized** to the API's slug format (`Shoe_Red 42` ->
  `shoe-red-42`). Two SKUs that normalize to the same slug collide (the second
  is treated as an update of the first). `visual_search_product_map` maps API SKUs back to products.
- **Only JPEG/PNG base images** are accepted by the API; products with a
  WEBP/GIF base image fail with a clear message.
- **Rate limits are per client IP.** Magento is the only client the API sees,
  so *all shoppers share one bucket* (default `/search` 20/min,
  `/products/import` 5/min). Raise `SEARCH_RATE_LIMIT` on the API for real
  traffic. Bulk batches wait out a `429` (`Retry-After`) up to twice.
- **Recommendations only cover synced products.** The API can only recommend
  products in its index, so sync your catalog first; unsynced products in a
  shopper's history are ignored.
- **A guest who logs in starts fresh** (guest cookie id vs. customer id).
  Erasing a customer's tracked history means calling
  `DELETE /shoppers/c<id>/events` on the API — the module does not do this
  automatically when a customer is deleted.
- Logs older than the retention setting (default 30 days) are pruned daily by cron.

## Tested / not tested

Verified: PHP, XML, JSON and JS syntax of every file. The HTTP client, the
shopper resolver, the event tracker and the event consumer were run (with
stubbed Magento classes) against the *real* API routes and a real pgvector
Postgres: event delivery, cold-start and personalized recommendations,
repeated `exclude_sku` encoding, error handling, and best-effort behavior
when the queue or API is down. **Not run inside a real Magento install** —
DI compilation, the admin grid/button/mass action, queue delivery, the
observers, and the storefront pages/JS (image search, "Recommended for you",
the view beacon) still need a smoke test.
