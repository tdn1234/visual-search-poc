<?php
declare(strict_types=1);

namespace VisualSearch\Connector\Block;

use Magento\Catalog\Helper\Data as CatalogHelper;
use Magento\Framework\View\Element\Template;
use Magento\Framework\View\Element\Template\Context;
use VisualSearch\Connector\Model\Config;

/**
 * Product-page hook for the "view" beacon (`js/track-view.js`).
 * Emits nothing visible; renders only when event tracking is on.
 */
class ProductTracker extends Template
{
    protected $_template = 'VisualSearch_Connector::product-tracker.phtml';

    public function __construct(
        Context $context,
        private readonly Config $config,
        private readonly CatalogHelper $catalogHelper,
        array $data = []
    ) {
        parent::__construct($context, $data);
    }

    /** JSON for the container's `data-mage-init`; empty if there's no current product. */
    public function getInitConfig(): string
    {
        $product = $this->catalogHelper->getProduct();
        if (!$product || !$product->getId()) {
            return '';
        }

        return (string)json_encode([
            'VisualSearch_Connector/js/track-view' => [
                'url' => $this->getUrl('visualsearch/event/track'),
                'productId' => (int)$product->getId(),
            ],
        ]);
    }

    protected function _toHtml(): string
    {
        return $this->config->isTrackingEnabled() && $this->getInitConfig() !== '' ? parent::_toHtml() : '';
    }
}
