<?php
declare(strict_types=1);

namespace VisualSearch\Connector\Block;

use Magento\Catalog\Helper\Data as CatalogHelper;
use Magento\Framework\View\Element\Template;
use Magento\Framework\View\Element\Template\Context;
use VisualSearch\Connector\Model\Config;

/**
 * Empty placeholder for the "Recommended for you" widget.
 *
 * The recommendations themselves are personal, so they can't be rendered
 * into a full-page-cached page; this block only outputs a container and the
 * config for `js/recommendations.js`, which fetches them per visitor.
 */
class Recommendations extends Template
{
    protected $_template = 'VisualSearch_Connector::recommendations.phtml';

    public function __construct(
        Context $context,
        private readonly Config $config,
        private readonly CatalogHelper $catalogHelper,
        array $data = []
    ) {
        parent::__construct($context, $data);
    }

    /** JSON for the container's `data-mage-init`. */
    public function getInitConfig(): string
    {
        $product = $this->catalogHelper->getProduct();

        return (string)json_encode([
            'VisualSearch_Connector/js/recommendations' => [
                'url' => $this->getUrl('visualsearch/recommendation/get'),
                // On a product page: lets the endpoint leave that product out of its own recommendations.
                'productId' => $product ? (int)$product->getId() : null,
                'title' => (string)__('Recommended for you'),
            ],
        ]);
    }

    protected function _toHtml(): string
    {
        return $this->config->areRecommendationsEnabled() ? parent::_toHtml() : '';
    }
}
