<?php
declare(strict_types=1);

namespace VisualSearch\Connector\Model\Api;

use GuzzleHttp\ClientFactory;
use GuzzleHttp\Exception\GuzzleException;
use VisualSearch\Connector\Model\Config;

/**
 * Thin HTTP client for the visual-search-poc API (backend/app/api/*.py).
 *
 * Knows the wire format only -- multipart field names, query params, the
 * X-API-Key header -- and returns a {@see Response} for *any* HTTP status so
 * callers decide what a 409 or 429 means. Only transport failures throw.
 *
 * A "payload" here is the array produced by
 * {@see \VisualSearch\Connector\Model\ProductPayloadBuilder::build()}.
 */
class Client
{
    public function __construct(
        private readonly Config $config,
        private readonly ClientFactory $clientFactory
    ) {
    }

    /**
     * POST /products -- synchronous; embeds the photo before responding.
     * 201 = created, 409 = sku already exists.
     *
     * @param array<string, mixed> $payload
     * @throws ApiException
     */
    public function createProduct(array $payload): Response
    {
        return $this->request('POST', '/products', ['multipart' => array_merge(
            $this->metaFields($payload['meta']),
            [$this->filePart('file', $payload)]
        )]);
    }

    /**
     * PUT /products/{sku} -- replaces an existing product's metadata and photo.
     * 200 = updated, 404 = sku not in the index.
     *
     * @param array<string, mixed> $payload
     * @throws ApiException
     */
    public function updateProduct(array $payload): Response
    {
        $meta = $payload['meta'];
        unset($meta['sku']);
        return $this->request('PUT', '/products/' . rawurlencode($payload['meta']['sku']), ['multipart' => array_merge(
            $this->metaFields($meta),
            [$this->filePart('file', $payload)]
        )]);
    }

    /**
     * POST /products/import -- asynchronous; 202 + batch_id, the service's
     * worker embeds each product later. `products[i]` pairs with `files[i]`.
     *
     * @param array<int, array<string, mixed>> $payloads (max 100 -- the API's batch cap)
     * @throws ApiException
     */
    public function importProducts(array $payloads): Response
    {
        $multipart = [[
            'name' => 'products',
            'contents' => (string)json_encode(array_map(
                static fn (array $payload): array => [
                    'sku' => $payload['meta']['sku'],
                    'name' => $payload['meta']['name'],
                    'price' => (float)$payload['meta']['price'],
                    'category' => $payload['meta']['category'],
                    'color' => $payload['meta']['color'] ?? null,
                ],
                $payloads
            )),
        ]];
        foreach ($payloads as $payload) {
            $multipart[] = $this->filePart('files', $payload);
        }
        return $this->request('POST', '/products/import', ['multipart' => $multipart]);
    }

    /**
     * POST /search -- top-K visually similar products for an uploaded image.
     *
     * @throws ApiException
     */
    public function searchByImage(
        string $imageBytes,
        string $mime,
        string $filename,
        bool $matchCategory,
        bool $matchColor
    ): Response {
        return $this->request('POST', '/search', [
            'query' => [
                'match_category' => $matchCategory ? 'true' : 'false',
                'match_color' => $matchColor ? 'true' : 'false',
            ],
            'multipart' => [[
                'name' => 'file',
                'filename' => $filename,
                'contents' => $imageBytes,
                'headers' => ['Content-Type' => $mime],
            ]],
        ]);
    }

    /**
     * POST /agent/chat -- one turn with the shopping assistant (multipart form).
     * Optional fields are omitted when empty. The LLM is slow, so the request gets its own timeout.
     *
     * @param array{bytes: string, mime: string, filename: string}|null $image
     * @throws ApiException
     */
    public function chat(string $message, ?string $shopperId, ?string $sessionId, ?array $image, int $timeout): Response
    {
        $multipart = [['name' => 'message', 'contents' => $message]];
        if ($shopperId !== null && $shopperId !== '') {
            $multipart[] = ['name' => 'shopper_id', 'contents' => $shopperId];
        }
        if ($sessionId !== null && $sessionId !== '') {
            $multipart[] = ['name' => 'session_id', 'contents' => $sessionId];
        }
        if ($image !== null) {
            $multipart[] = [
                'name' => 'file',
                'filename' => $image['filename'],
                'contents' => $image['bytes'],
                'headers' => ['Content-Type' => $image['mime']],
            ];
        }
        return $this->request('POST', '/agent/chat', ['multipart' => $multipart, 'timeout' => $timeout]);
    }

    /**
     * POST /events -- record shopper behavior (view / add_to_cart / purchase).
     *
     * @param array<int, array{shopper_id: string, sku: string, event_type: string, occurred_at?: string}> $events
     *        1-100 events
     * @throws ApiException
     */
    public function trackEvents(array $events): Response
    {
        return $this->request('POST', '/events', ['json' => ['events' => array_values($events)]]);
    }

    /**
     * GET /recommendations -- personalized products for one shopper.
     * Response strategy is "personalized", "popular" (cold start) or "none".
     *
     * @param string[] $excludeSkus API skus never to return (e.g. the product page being viewed)
     * @throws ApiException
     */
    public function getRecommendations(string $shopperId, int $limit, array $excludeSkus = []): Response
    {
        // FastAPI reads a list as a *repeated* parameter (exclude_sku=a&exclude_sku=b), but
        // Guzzle's array query support would send exclude_sku[0]=a -- so build the string by hand.
        $query = http_build_query(['shopper_id' => $shopperId, 'limit' => $limit]);
        foreach ($excludeSkus as $sku) {
            $query .= '&exclude_sku=' . rawurlencode($sku);
        }
        return $this->request('GET', '/recommendations', ['query' => $query]);
    }

    /**
     * @param array<string, mixed> $options Guzzle request options
     * @throws ApiException
     */
    private function request(string $method, string $path, array $options): Response
    {
        $baseUrl = $this->config->getApiUrl();
        if ($baseUrl === '') {
            throw new ApiException('Visual search API URL is not configured.');
        }

        $client = $this->clientFactory->create(['config' => [
            'base_uri' => $baseUrl . '/',
            'timeout' => $this->config->getTimeout(),
            'connect_timeout' => 5,
            'http_errors' => false,
            // A followed redirect turns the POST into a GET (a bogus 200); surface it instead.
            'allow_redirects' => false,
            'headers' => ['X-API-Key' => $this->config->getApiKey(), 'Accept' => 'application/json'],
        ]]);

        try {
            $httpResponse = $client->request($method, ltrim($path, '/'), $options);
        } catch (GuzzleException $e) {
            throw new ApiException('Visual search API is unreachable: ' . $e->getMessage(), 0, $e);
        }

        $decoded = json_decode((string)$httpResponse->getBody(), true);
        $retryAfter = $httpResponse->getHeaderLine('Retry-After');

        return new Response(
            $httpResponse->getStatusCode(),
            is_array($decoded) ? $decoded : [],
            ctype_digit($retryAfter) ? (int)$retryAfter : null
        );
    }

    /**
     * @param array<string, mixed> $meta
     * @return array<int, array{name: string, contents: string}>
     */
    private function metaFields(array $meta): array
    {
        $fields = [];
        foreach (['sku', 'name', 'price', 'category', 'color'] as $key) {
            // Optional `color` is omitted rather than sent empty.
            if (($meta[$key] ?? null) !== null && $meta[$key] !== '') {
                $fields[] = ['name' => $key, 'contents' => (string)$meta[$key]];
            }
        }
        return $fields;
    }

    /**
     * @param array<string, mixed> $payload
     * @return array<string, mixed>
     */
    private function filePart(string $fieldName, array $payload): array
    {
        return [
            'name' => $fieldName,
            'filename' => $payload['filename'],
            'contents' => $payload['image'],
            'headers' => ['Content-Type' => $payload['mime']],
        ];
    }
}
