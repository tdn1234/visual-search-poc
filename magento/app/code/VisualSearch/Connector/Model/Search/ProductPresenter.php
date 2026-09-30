<?php
declare(strict_types=1);

namespace VisualSearch\Connector\Model\Search;

use Magento\Catalog\Helper\Image as ImageHelper;
use Magento\Catalog\Model\Product;
use Magento\Framework\Pricing\Helper\Data as PricingHelper;

/**
 * The bits of a product a storefront card needs (link, image, price),
 * shared by the image-search page and the recommendations widget.
 */
class ProductPresenter
{
    public function __construct(
        private readonly ImageHelper $imageHelper,
        private readonly PricingHelper $pricingHelper
    ) {
    }

    public function getImageUrl(Product $product): string
    {
        return $this->imageHelper->init($product, 'category_page_grid')->getUrl();
    }

    /** Formatted with the store currency, e.g. "$99.00" (plain text, not HTML). */
    public function getFormattedPrice(Product $product): string
    {
        $price = $product->getData('minimal_price') ?? $product->getFinalPrice();
        return $this->pricingHelper->currency((float)$price, true, false);
    }

    /**
     * @return array{id: int, name: string, url: string, image: string, price: string}
     */
    public function toArray(Product $product): array
    {
        return [
            'id' => (int)$product->getId(),
            'name' => (string)$product->getName(),
            'url' => (string)$product->getProductUrl(),
            'image' => $this->getImageUrl($product),
            'price' => $this->getFormattedPrice($product),
        ];
    }
}
