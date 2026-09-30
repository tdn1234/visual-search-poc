<?php
declare(strict_types=1);

namespace VisualSearch\Connector\Model\Search;

use Magento\Catalog\Model\Product;
use Magento\Catalog\Model\Product\Attribute\Source\Status as ProductStatus;
use Magento\Catalog\Model\Product\Visibility;
use Magento\Catalog\Model\ResourceModel\Product\CollectionFactory;
use Magento\Framework\Exception\LocalizedException;
use Psr\Log\LoggerInterface;
use VisualSearch\Connector\Model\Api\ApiException;
use VisualSearch\Connector\Model\Api\Client;
use VisualSearch\Connector\Model\Config;
use VisualSearch\Connector\Model\ResourceModel\ProductMap;

/**
 * Storefront "search by image": uploaded image -> visual search API -> the
 * matching *Magento* products, ranked by the API's similarity score.
 *
 * The API ranks over everything that was ever synced, so results are
 * re-checked against the storefront (enabled, visible in search, in this
 * store) -- a hit that's no longer buyable is dropped rather than shown.
 */
class ImageSearch
{
    public function __construct(
        private readonly Config $config,
        private readonly Client $client,
        private readonly ProductMap $productMap,
        private readonly CollectionFactory $collectionFactory,
        private readonly ProductStatus $productStatus,
        private readonly Visibility $productVisibility,
        private readonly LoggerInterface $logger
    ) {
    }

    /**
     * @return array<int, array{product: Product, score: float}> best match first
     * @throws LocalizedException with a shopper-safe message if the search can't run
     */
    public function search(string $imageBytes, string $mime, string $filename): array
    {
        try {
            $response = $this->client->searchByImage(
                $imageBytes,
                $mime,
                $filename,
                $this->config->shouldMatchCategory(),
                $this->config->shouldMatchColor()
            );
        } catch (ApiException $e) {
            $this->logger->error('Visual search unreachable: ' . $e->getMessage());
            throw new LocalizedException(__('Image search is temporarily unavailable. Please try again later.'));
        }

        // 503 = the service's index is empty (nothing synced yet): a valid "no results".
        if ($response->getStatusCode() === 503) {
            return [];
        }
        if (!$response->isSuccess()) {
            $this->logger->error('Visual search API error: ' . $response->getErrorMessage());
            throw new LocalizedException(
                $response->getStatusCode() === 400
                    ? __('We could not read that image. Please try a different JPEG, PNG or WEBP photo.')
                    : __('Image search is temporarily unavailable. Please try again later.')
            );
        }

        $scoreByApiSku = [];
        foreach ($response->getData()['results'] ?? [] as $hit) {
            $scoreByApiSku[(string)$hit['sku']] = (float)$hit['score'];
        }

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

        // Keep the API's ranking (its result order), not the collection's.
        $results = [];
        foreach ($scoreByApiSku as $apiSku => $score) {
            $productId = $productIdByApiSku[$apiSku] ?? null;
            if ($productId !== null && isset($productsById[$productId])) {
                $results[] = ['product' => $productsById[$productId], 'score' => $score];
            }
        }
        return $results;
    }
}
