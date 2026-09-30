<?php
declare(strict_types=1);

namespace VisualSearch\Connector\Observer;

use Magento\Framework\Event\Observer;
use Magento\Framework\Event\ObserverInterface;
use VisualSearch\Connector\Model\EventTracker;

/**
 * `sales_order_place_after`: every product in a placed order is a purchase.
 * The order's own customer wins over whoever the session says is browsing.
 */
class OrderPlaceObserver implements ObserverInterface
{
    public function __construct(private readonly EventTracker $tracker)
    {
    }

    public function execute(Observer $observer): void
    {
        /** @var \Magento\Sales\Model\Order|null $order */
        $order = $observer->getEvent()->getData('order');
        if ($order === null) {
            return;
        }

        $productIds = [];
        // Visible items = what the shopper bought (a configurable's child rows are skipped).
        foreach ($order->getAllVisibleItems() as $item) {
            $productIds[] = (int)$item->getProductId();
        }

        $customerId = (int)$order->getCustomerId();
        $this->tracker->track(
            EventTracker::PURCHASE,
            array_values(array_unique(array_filter($productIds))),
            $customerId > 0 ? 'c' . $customerId : null
        );
    }
}
