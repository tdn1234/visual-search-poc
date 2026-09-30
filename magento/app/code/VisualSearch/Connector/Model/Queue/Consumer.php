<?php
declare(strict_types=1);

namespace VisualSearch\Connector\Model\Queue;

use Magento\Framework\Serialize\Serializer\Json;
use Psr\Log\LoggerInterface;
use VisualSearch\Connector\Model\SyncService;

/**
 * Handler for the `visualsearch.product.sync` queue.
 *
 * Run with:  bin/magento queue:consumers:start visualsearch.product.sync
 * (or let Magento's cron consumers_runner pick it up).
 */
class Consumer
{
    public function __construct(
        private readonly SyncService $syncService,
        private readonly Json $json,
        private readonly LoggerInterface $logger
    ) {
    }

    public function process(string $message): void
    {
        try {
            $data = $this->json->unserialize($message);
            $this->syncService->processQueued($data['items'] ?? []);
        } catch (\Throwable $e) {
            // Deliberately swallowed: rethrowing would make the DB queue retry a
            // poison message forever. Expected failures are already on the log rows.
            $this->logger->error('Visual search queue message failed: ' . $e->getMessage(), ['exception' => $e]);
        }
    }
}
