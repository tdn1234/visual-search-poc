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
| `success` | Created in the index (`201`, single sync) |
| `skipped` | Already in the index (`409`) |
| `failed` | Product couldn't be built (no price / no JPEG-PNG image), API error, or API unreachable -- see Message |

## Things to know

- **The API has no update or delete endpoint.** Re-saving an already-synced
  product logs `skipped` (409) and the index keeps the old image/metadata.
  Products deleted in Magento stay in the index; storefront results are
  re-checked against Magento, so they're simply not shown.
- **SKUs are normalized** to the API's slug format (`Shoe_Red 42` ->
  `shoe-red-42`). Two SKUs that normalize to the same slug collide (the second
  is `skipped`). `visual_search_product_map` maps API SKUs back to products.
- **Only JPEG/PNG base images** are accepted by the API; products with a
  WEBP/GIF base image fail with a clear message.
- **Rate limits are per client IP.** Magento is the only client the API sees,
  so *all shoppers share one bucket* (default `/search` 20/min,
  `/products/import` 5/min). Raise `SEARCH_RATE_LIMIT` on the API for real
  traffic. Bulk batches wait out a `429` (`Retry-After`) up to twice.
- Logs older than the retention setting (default 30 days) are pruned daily by cron.

## Tested / not tested

Verified: PHP + XML + JSON syntax of every file, and the HTTP client
(`Model/Api/Client`) against a mock server with the API's exact route
signatures (201/409/400/401/202 handling, multipart field names, search
flags, unreachable host). **Not run inside a real Magento install** -- DI
compilation, the admin grid/button/mass action, queue delivery, and the
storefront pages still need a smoke test.
