<?php
declare(strict_types=1);

namespace VisualSearch\Connector\Observer;

use Magento\Framework\Event\Observer;
use Magento\Framework\Event\ObserverInterface;
use VisualSearch\Connector\Model\EventTracker;

/**
 * `checkout_cart_add_product_complete`: the shopper added a product to their cart.
 * (Cart adds aren't full-page-cached, so a server-side observer sees every one.)
 */
class CartAddObserver implements ObserverInterface
{
    public function __construct(private readonly EventTracker $tracker)
    {
    }

    public function execute(Observer $observer): void
    {
        $product = $observer->getEvent()->getData('product');
        if ($product !== null && (int)$product->getId() > 0) {
            $this->tracker->track(EventTracker::ADD_TO_CART, [(int)$product->getId()]);
        }
    }
}
