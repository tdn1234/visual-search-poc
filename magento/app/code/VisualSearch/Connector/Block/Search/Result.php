<?php
declare(strict_types=1);

namespace VisualSearch\Connector\Block\Search;

use Magento\Catalog\Helper\Image as ImageHelper;
use Magento\Catalog\Model\Product;
use Magento\Framework\Pricing\Helper\Data as PricingHelper;
use Magento\Framework\View\Element\Template;
use Magento\Framework\View\Element\Template\Context;
use VisualSearch\Connector\Model\Config;
use VisualSearch\Connector\Model\Search\ResultHolder;

/**
 * Product cards for the image-search results page.
 */
class Result extends Template
{
    protected $_template = 'VisualSearch_Connector::search/result.phtml';

    public function __construct(
        Context $context,
        private readonly Config $config,
        private readonly ResultHolder $resultHolder,
        private readonly ImageHelper $imageHelper,
        private readonly PricingHelper $pricingHelper,
        array $data = []
    ) {
        parent::__construct($context, $data);
    }

    /**
     * @return array<int, array{product: Product, score: float}>|null null = no search was run
     */
    public function getResults(): ?array
    {
        return $this->resultHolder->getResults();
    }

    public function getError(): ?string
    {
        return $this->resultHolder->getError();
    }

    public function getImageUrl(Product $product): string
    {
        return $this->imageHelper->init($product, 'category_page_grid')->getUrl();
    }

    public function getPriceHtml(Product $product): string
    {
        $price = $product->getData('minimal_price') ?? $product->getFinalPrice();
        return $this->pricingHelper->currency((float)$price, true, false);
    }

    /** Cosine similarity in [-1, 1] shown as a 0-100 "match" percentage. */
    public function getMatchPercent(float $score): int
    {
        return (int)round(max(0.0, min(1.0, $score)) * 100);
    }

    protected function _toHtml(): string
    {
        return $this->config->isEnabled() ? parent::_toHtml() : '';
    }
}
