<?php
declare(strict_types=1);

namespace VisualSearch\Connector\Observer;

use Magento\Framework\Event\Observer;
use Magento\Framework\Event\ObserverInterface;
use Psr\Log\LoggerInterface;
use VisualSearch\Connector\Model\Config;
use VisualSearch\Connector\Model\Source\AutoSyncMode;
use VisualSearch\Connector\Model\Source\SyncType;
use VisualSearch\Connector\Model\SyncService;

/**
 * Syncs a product to the visual search service after it is saved.
 *
 * Must never break the save: the product is already committed by the time
 * this runs, and an unreachable API is logged (sync log) rather than shown
 * as an admin error.
 */
class ProductSaveObserver implements ObserverInterface
{
    /** @var array<int, true> product ids already handled in this request */
    private array $handled = [];

    public function __construct(
        private readonly Config $config,
        private readonly SyncService $syncService,
        private readonly LoggerInterface $logger
    ) {
    }

    public function execute(Observer $observer): void
    {
        if (!$this->config->isAutoSyncEnabled()) {
            return;
        }

        $productId = (int)$observer->getEvent()->getProduct()->getId();
        // A single admin save can dispatch this event more than once.
        if ($productId === 0 || isset($this->handled[$productId])) {
            return;
        }
        $this->handled[$productId] = true;

        try {
            if ($this->config->getAutoSyncMode() === AutoSyncMode::QUEUE) {
                $this->syncService->enqueue([$productId], SyncType::AUTO);
            } else {
                $this->syncService->syncNow($productId, SyncType::AUTO);
            }
        } catch (\Throwable $e) {
            $this->logger->error('Visual search on-save sync failed: ' . $e->getMessage(), ['exception' => $e]);
        }
    }
}
