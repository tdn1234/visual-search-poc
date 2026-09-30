<?php
declare(strict_types=1);

namespace VisualSearch\Connector\Model;

use Magento\Catalog\Model\ResourceModel\Product as ProductResource;
use Magento\Framework\MessageQueue\PublisherInterface;
use Magento\Framework\Serialize\Serializer\Json;
use Psr\Log\LoggerInterface;
use VisualSearch\Connector\Model\Shopper\ShopperResolver;

/**
 * Reports shopper behavior to the visual search service -- via the Magento
 * queue, never inline, so a slow or unreachable API can't slow down (or break)
 * a product page, add-to-cart, or checkout. Publishing is one cheap DB insert.
 *
 * Everything here is best-effort: tracking failures are logged and swallowed.
 */
class EventTracker
{
    public const TOPIC = 'visualsearch.event.track';

    public const VIEW = 'view';
    public const ADD_TO_CART = 'add_to_cart';
    public const PURCHASE = 'purchase';

    public function __construct(
        private readonly Config $config,
        private readonly ShopperResolver $shopperResolver,
        private readonly ProductResource $productResource,
        private readonly PublisherInterface $publisher,
        private readonly Json $json,
        private readonly LoggerInterface $logger
    ) {
    }

    /**
     * @param int[] $productIds Magento product entity ids
     * @param string|null $shopperId override (e.g. the order's own customer); default: the current shopper
     */
    public function track(string $eventType, array $productIds, ?string $shopperId = null): void
    {
        if (!$this->config->isTrackingEnabled() || $productIds === []) {
            return;
        }

        try {
            $shopperId ??= $this->shopperResolver->resolve();
            $occurredAt = gmdate('c');

            // Look the SKU up by id rather than trusting an order/quote item's own sku field:
            // for configurable products those can hold the *child's* sku.
            $events = [];
            foreach ($this->productResource->getProductsSku(array_map('intval', $productIds)) as $row) {
                $events[] = [
                    'shopper_id' => $shopperId,
                    'sku' => ProductPayloadBuilder::toApiSku((string)$row['sku'], (int)$row['entity_id']),
                    'event_type' => $eventType,
                    'occurred_at' => $occurredAt,
                ];
            }

            if ($events !== []) {
                $this->publisher->publish(self::TOPIC, $this->json->serialize(['events' => $events]));
            }
        } catch (\Throwable $e) {
            $this->logger->warning('Visual search event tracking failed: ' . $e->getMessage());
        }
    }
}
