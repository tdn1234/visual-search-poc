<?php
declare(strict_types=1);

namespace VisualSearch\Connector\Model\Search;

use Magento\Catalog\Model\Product;
use Magento\Catalog\Model\Product\Attribute\Source\Status as ProductStatus;
use Magento\Catalog\Model\Product\Visibility;
use Magento\Catalog\Model\ResourceModel\Product\CollectionFactory;
use VisualSearch\Connector\Model\ResourceModel\ProductMap;

/**
 * Turns the visual search API's SKUs back into *storefront* products.
 *
 * The API ranks over everything that was ever synced, so its hits are
 * re-checked against Magento (enabled, visible in search, in this store):
 * a hit that's no longer buyable is dropped rather than shown. Shared by
 * image search and recommendations.
 */
class ProductResolver
{
    public function __construct(
        private readonly ProductMap $productMap,
        private readonly CollectionFactory $collectionFactory,
        private readonly ProductStatus $productStatus,
        private readonly Visibility $productVisibility
    ) {
    }

    /**
     * @param array<string, float> $scoreByApiSku API sku => score, in the API's ranking order
     * @return array<int, array{product: Product, score: float}> same order, unresolvable hits dropped
     */
    public function resolve(array $scoreByApiSku): array
    {
        $productIdByApiSku = $this->productMap->getProductIdsByApiSkus(array_keys($scoreByApiSku));
        if ($productIdByApiSku === []) {
            return [];
        }

        $collection = $this->collectionFactory->create();
        $collection->addAttributeToSelect(['name', 'image', 'small_image', 'thumbnail'])
            ->addIdFilter(array_values($productIdByApiSku))
            ->addStoreFilter()
            ->addAttributeToFilter('status', ['in' => $this->productStatus->getVisibleStatusIds()])
            ->setVisibility($this->productVisibility->getVisibleInSearchIds())
            ->addPriceData()
            ->addUrlRewrite();

        $productsById = [];
        foreach ($collection as $product) {
            $productsById[(int)$product->getId()] = $product;
        }

        // Keep the API's ranking (its order), not the collection's.
        $results = [];
        foreach ($scoreByApiSku as $apiSku => $score) {
            $productId = $productIdByApiSku[$apiSku] ?? null;
            if ($productId !== null && isset($productsById[$productId])) {
                $results[] = ['product' => $productsById[$productId], 'score' => (float)$score];
            }
        }
        return $results;
    }
}
