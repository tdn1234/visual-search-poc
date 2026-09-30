<?php
declare(strict_types=1);

namespace VisualSearch\Connector\Controller\Event;

use Magento\Framework\App\Action\HttpPostActionInterface;
use Magento\Framework\App\RequestInterface;
use Magento\Framework\Controller\Result\Json;
use Magento\Framework\Controller\Result\JsonFactory;
use VisualSearch\Connector\Model\EventTracker;

/**
 * POST /visualsearch/event/track -- the product-page "view" beacon.
 *
 * Views have to be reported from the browser: with full-page cache on, a
 * product page is usually served straight from the cache and never reaches
 * PHP, so a server-side observer would miss most views. (Add-to-cart and
 * purchase aren't cached, so those are ordinary observers.)
 *
 * Only `view` is accepted here, and Magento's storefront CSRF check (the
 * `form_key` the beacon sends) applies automatically.
 */
class Track implements HttpPostActionInterface
{
    public function __construct(
        private readonly RequestInterface $request,
        private readonly JsonFactory $jsonFactory,
        private readonly EventTracker $tracker
    ) {
    }

    public function execute(): Json
    {
        $result = $this->jsonFactory->create();

        $productId = (int)$this->request->getParam('product_id');
        if ($productId <= 0) {
            return $result->setHttpResponseCode(400)->setData(['ok' => false]);
        }

        // No-ops (and still answers ok) if tracking is disabled -- nothing for the browser to act on.
        $this->tracker->track(EventTracker::VIEW, [$productId]);

        return $result->setData(['ok' => true]);
    }
}
