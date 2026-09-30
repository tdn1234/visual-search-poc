<?php
declare(strict_types=1);

namespace VisualSearch\Connector\Model\Search;

use Magento\Catalog\Model\Product;
use Magento\Framework\Exception\LocalizedException;
use Psr\Log\LoggerInterface;
use VisualSearch\Connector\Model\Api\ApiException;
use VisualSearch\Connector\Model\Api\Client;
use VisualSearch\Connector\Model\Config;

/**
 * Storefront "search by image": uploaded image -> visual search API -> the
 * matching *Magento* products, ranked by the API's similarity score.
 * (Mapping API SKUs back to buyable products is {@see ProductResolver}'s job.)
 */
class ImageSearch
{
    public function __construct(
        private readonly Config $config,
        private readonly Client $client,
        private readonly ProductResolver $productResolver,
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

        return $this->productResolver->resolve($scoreByApiSku);
    }
}
