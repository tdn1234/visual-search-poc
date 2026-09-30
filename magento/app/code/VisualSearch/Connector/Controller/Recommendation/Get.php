<?php
declare(strict_types=1);

namespace VisualSearch\Connector\Controller\Recommendation;

use Magento\Catalog\Model\ResourceModel\Product as ProductResource;
use Magento\Framework\App\Action\HttpGetActionInterface;
use Magento\Framework\App\RequestInterface;
use Magento\Framework\Controller\Result\Json;
use Magento\Framework\Controller\Result\JsonFactory;
use Psr\Log\LoggerInterface;
use VisualSearch\Connector\Model\Api\ApiException;
use VisualSearch\Connector\Model\Api\Client;
use VisualSearch\Connector\Model\Config;
use VisualSearch\Connector\Model\ProductPayloadBuilder;
use VisualSearch\Connector\Model\Search\ProductPresenter;
use VisualSearch\Connector\Model\Search\ProductResolver;
use VisualSearch\Connector\Model\Shopper\ShopperResolver;

/**
 * GET /visualsearch/recommendation/get -- "Recommended for you" data as JSON.
 *
 * Deliberately a JSON endpoint the page fetches with AJAX, not a block
 * rendered into the page: the page itself is full-page-cached and shared by
 * every visitor, while these products are personal. A JSON response is
 * never stored by the page cache, and is sent no-store as well.
 *
 * Never errors to the browser: any failure (API down, disabled, nothing to
 * recommend) is an empty list, and the widget simply doesn't render.
 */
class Get implements HttpGetActionInterface
{
    public function __construct(
        private readonly RequestInterface $request,
        private readonly JsonFactory $jsonFactory,
        private readonly Config $config,
        private readonly Client $client,
        private readonly ShopperResolver $shopperResolver,
        private readonly ProductResource $productResource,
        private readonly ProductResolver $productResolver,
        private readonly ProductPresenter $presenter,
        private readonly LoggerInterface $logger
    ) {
    }

    public function execute(): Json
    {
        $result = $this->jsonFactory->create();
        $result->setHeader('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0', true);

        return $result->setData(['items' => $this->loadItems()]);
    }

    /**
     * @return array<int, array{id: int, name: string, url: string, image: string, price: string}>
     */
    private function loadItems(): array
    {
        if (!$this->config->areRecommendationsEnabled()) {
            return [];
        }

        try {
            $response = $this->client->getRecommendations(
                $this->shopperResolver->resolve(),
                $this->config->getRecommendationLimit(),
                $this->excludedApiSkus()
            );
            if (!$response->isSuccess()) {
                $this->logger->warning('Visual search recommendations failed: ' . $response->getErrorMessage());
                return [];
            }

            $scoreByApiSku = [];
            foreach ($response->getData()['results'] ?? [] as $hit) {
                $scoreByApiSku[(string)$hit['sku']] = (float)$hit['score'];
            }

            return array_map(
                fn (array $hit): array => $this->presenter->toArray($hit['product']),
                $this->productResolver->resolve($scoreByApiSku)
            );
        } catch (ApiException $e) {
            $this->logger->warning('Visual search recommendations unavailable: ' . $e->getMessage());
        } catch (\Throwable $e) {
            $this->logger->error('Visual search recommendations error: ' . $e->getMessage(), ['exception' => $e]);
        }
        return [];
    }

    /** On a product page, never recommend the product being looked at. */
    private function excludedApiSkus(): array
    {
        $productId = (int)$this->request->getParam('product_id');
        if ($productId <= 0) {
            return [];
        }
        return array_map(
            static fn (array $row): string => ProductPayloadBuilder::toApiSku((string)$row['sku'], (int)$row['entity_id']),
            $this->productResource->getProductsSku([$productId])
        );
    }
}
