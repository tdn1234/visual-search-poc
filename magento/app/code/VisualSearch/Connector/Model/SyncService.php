<?php
declare(strict_types=1);

namespace VisualSearch\Connector\Model;

use Magento\Catalog\Api\ProductRepositoryInterface;
use Magento\Catalog\Model\ResourceModel\Product\CollectionFactory as ProductCollectionFactory;
use Magento\Framework\Exception\LocalizedException;
use Magento\Framework\MessageQueue\PublisherInterface;
use Magento\Framework\Serialize\Serializer\Json;
use Psr\Log\LoggerInterface;
use VisualSearch\Connector\Model\Api\ApiException;
use VisualSearch\Connector\Model\Api\Client;
use VisualSearch\Connector\Model\Api\Response;
use VisualSearch\Connector\Model\ResourceModel\ProductMap;
use VisualSearch\Connector\Model\ResourceModel\SyncLog as SyncLogResource;
use VisualSearch\Connector\Model\Source\Status;
use VisualSearch\Connector\Model\Source\SyncType;

/**
 * Orchestrates every product sync and writes the sync log.
 *
 *  - {@see syncNow()}: one product, synchronously, via POST /products
 *    (manual button, and on-save in "immediate" mode).
 *  - {@see enqueue()} + {@see processQueued()}: many products through the
 *    Magento queue, sent to the API in batches via POST /products/import
 *    (product-grid mass action, and on-save in "queue" mode).
 *
 * The two paths report different terminal states because the API differs:
 * POST /products answers with the real outcome (success/skipped/failed),
 * POST /products/import only says "accepted" (status `queued`) -- the
 * service has no status endpoint, so per-item results live in its worker logs.
 */
class SyncService
{
    public const TOPIC = 'visualsearch.product.sync';

    /** How many times a batch waits out an API rate limit (429) before giving up. */
    private const RATE_LIMIT_RETRIES = 2;
    private const MAX_RETRY_WAIT_SECONDS = 65;

    public function __construct(
        private readonly Config $config,
        private readonly Client $client,
        private readonly ProductPayloadBuilder $payloadBuilder,
        private readonly ProductRepositoryInterface $productRepository,
        private readonly ProductCollectionFactory $productCollectionFactory,
        private readonly SyncLogResource $syncLog,
        private readonly ProductMap $productMap,
        private readonly PublisherInterface $publisher,
        private readonly Json $json,
        private readonly LoggerInterface $logger
    ) {
    }

    /**
     * Sync one product right now and log the outcome.
     *
     * @return array{status: string, message: string}
     */
    public function syncNow(int $productId, string $syncType): array
    {
        $startedAt = microtime(true);
        $sku = '';
        $apiSku = null;
        $httpStatus = null;

        try {
            $product = $this->loadProduct($productId);
            $sku = (string)$product->getSku();
            $payload = $this->payloadBuilder->build($product);
            $apiSku = $payload['meta']['sku'];

            $response = $this->client->createProduct($payload);
            $httpStatus = $response->getStatusCode();
            [$status, $message] = $this->interpretCreateResponse($response);

            if ($status === Status::SUCCESS || $status === Status::SKIPPED) {
                $this->productMap->upsert($productId, $apiSku);
            }
        } catch (LocalizedException $e) {
            [$status, $message] = [Status::FAILED, $e->getMessage()];
        } catch (ApiException $e) {
            [$status, $message] = [Status::FAILED, $e->getMessage()];
        } catch (\Throwable $e) {
            $this->logger->error('Visual search sync failed unexpectedly', ['exception' => $e]);
            [$status, $message] = [Status::FAILED, 'Unexpected error: ' . $e->getMessage()];
        }

        $this->syncLog->createEntry([
            'product_id' => $productId,
            'sku' => $sku !== '' ? $sku : (string)$productId,
            'api_sku' => $apiSku,
            'sync_type' => $syncType,
            'status' => $status,
            'http_status' => $httpStatus,
            'message' => $message,
            'duration_ms' => (int)round((microtime(true) - $startedAt) * 1000),
        ]);

        return ['status' => $status, 'message' => $message];
    }

    /**
     * Log each product as `pending` and publish them to the Magento queue in
     * batches of the configured size. Cheap: no API call, no image reads.
     *
     * @param int[] $productIds
     * @return int number of products queued
     */
    public function enqueue(array $productIds, string $syncType = SyncType::BULK): int
    {
        $productIds = array_values(array_unique(array_map('intval', $productIds)));
        if ($productIds === []) {
            return 0;
        }

        $skus = [];
        $collection = $this->productCollectionFactory->create();
        $collection->addFieldToFilter('entity_id', ['in' => $productIds]);
        foreach ($collection as $product) {
            $skus[(int)$product->getId()] = (string)$product->getSku();
        }

        $items = [];
        foreach ($productIds as $productId) {
            if (!isset($skus[$productId])) {
                continue;
            }
            $items[] = [
                'product_id' => $productId,
                'log_id' => $this->syncLog->createEntry([
                    'product_id' => $productId,
                    'sku' => $skus[$productId],
                    'api_sku' => ProductPayloadBuilder::toApiSku($skus[$productId], $productId),
                    'sync_type' => $syncType,
                    'status' => Status::PENDING,
                    'message' => 'Waiting in the Magento queue.',
                ]),
            ];
        }

        foreach (array_chunk($items, $this->config->getChunkSize()) as $chunk) {
            $this->publisher->publish(self::TOPIC, $this->json->serialize(['items' => $chunk]));
        }

        return count($items);
    }

    /**
     * Queue consumer entry point: send one batch of `enqueue()`d products to
     * POST /products/import and update their log rows.
     *
     * Never throws for expected failures -- they're recorded on the log rows
     * instead, so a bad batch doesn't get redelivered forever.
     *
     * @param array<int, array{product_id: int, log_id: int}> $items
     */
    public function processQueued(array $items): void
    {
        if (!$this->config->isEnabled()) {
            foreach ($items as $item) {
                $this->fail((int)$item['log_id'], 'Visual Search module is disabled.');
            }
            return;
        }

        // 1. Build a payload per product; ones that can't be built fail individually.
        $payloadsByLogId = [];
        $productIdsByLogId = [];
        foreach ($items as $item) {
            $logId = (int)$item['log_id'];
            try {
                $product = $this->loadProduct((int)$item['product_id']);
                $payloadsByLogId[$logId] = $this->payloadBuilder->build($product);
                $productIdsByLogId[$logId] = (int)$item['product_id'];
            } catch (LocalizedException $e) {
                $this->fail($logId, $e->getMessage());
            } catch (\Throwable $e) {
                $this->logger->error('Visual search payload build failed', ['exception' => $e]);
                $this->fail($logId, 'Unexpected error: ' . $e->getMessage());
            }
        }
        if ($payloadsByLogId === []) {
            return;
        }

        // 2. One import request for the whole batch.
        $startedAt = microtime(true);
        try {
            $response = $this->sendBatch(array_values($payloadsByLogId));
        } catch (ApiException $e) {
            foreach (array_keys($payloadsByLogId) as $logId) {
                $this->fail($logId, $e->getMessage());
            }
            return;
        }
        $durationMs = (int)round((microtime(true) - $startedAt) * 1000);

        // 3. Record the result on every product in the batch.
        if ($response->getStatusCode() === 202) {
            $batchId = (string)($response->getData()['batch_id'] ?? '');
            foreach ($payloadsByLogId as $logId => $payload) {
                $this->productMap->upsert($productIdsByLogId[$logId], $payload['meta']['sku']);
                $this->syncLog->updateEntry($logId, [
                    'status' => Status::QUEUED,
                    'http_status' => 202,
                    'batch_id' => $batchId !== '' ? $batchId : null,
                    'duration_ms' => $durationMs,
                    'message' => 'Accepted by the visual search import queue. The service processes it '
                        . 'asynchronously; per-product results are in its worker log'
                        . ($batchId !== '' ? " (grep {$batchId})." : '.'),
                ]);
            }
            return;
        }

        $message = $this->describeFailure($response);
        foreach (array_keys($payloadsByLogId) as $logId) {
            $this->syncLog->updateEntry($logId, [
                'status' => Status::FAILED,
                'http_status' => $response->getStatusCode(),
                'duration_ms' => $durationMs,
                'message' => $message,
            ]);
        }
    }

    /**
     * POST /products/import is limited to a few calls per minute per client
     * IP, so a 429 is waited out (Retry-After) a couple of times -- this runs
     * in a queue consumer, where sleeping is fine.
     *
     * @param array<int, array<string, mixed>> $payloads
     * @throws ApiException
     */
    private function sendBatch(array $payloads): Response
    {
        $attempt = 0;
        while (true) {
            $response = $this->client->importProducts($payloads);
            if ($response->getStatusCode() !== 429 || $attempt >= self::RATE_LIMIT_RETRIES) {
                return $response;
            }
            $attempt++;
            sleep(min($response->getRetryAfter() ?? 30, self::MAX_RETRY_WAIT_SECONDS));
        }
    }

    /**
     * @return array{0: string, 1: string} [status, message]
     */
    private function interpretCreateResponse(Response $response): array
    {
        return match ($response->getStatusCode()) {
            201 => [Status::SUCCESS, 'Product created in the visual search index.'],
            409 => [Status::SKIPPED, 'Already in the visual search index. The API has no update endpoint, '
                . 'so the existing entry was kept.'],
            default => [Status::FAILED, $this->describeFailure($response)],
        };
    }

    private function describeFailure(Response $response): string
    {
        return match ($response->getStatusCode()) {
            401 => 'Visual search API rejected the API key (401). Check Stores > Configuration > Services > Visual Search.',
            429 => 'Visual search API rate limit exceeded (429). Try again in a minute.',
            default => sprintf('API error %d: %s', $response->getStatusCode(), $response->getErrorMessage()),
        };
    }

    private function fail(int $logId, string $message): void
    {
        $this->syncLog->updateEntry($logId, ['status' => Status::FAILED, 'message' => $message]);
    }

    private function loadProduct(int $productId): \Magento\Catalog\Model\Product
    {
        // Store 0 (admin values) and forced reload: never sync a stale or store-view-specific copy.
        /** @var \Magento\Catalog\Model\Product $product */
        $product = $this->productRepository->getById($productId, false, 0, true);
        return $product;
    }
}
