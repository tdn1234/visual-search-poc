<?php
declare(strict_types=1);

namespace VisualSearch\Connector\Model\Chat;

use Magento\Framework\Exception\LocalizedException;
use Psr\Log\LoggerInterface;
use VisualSearch\Connector\Model\Api\ApiException;
use VisualSearch\Connector\Model\Api\Client;
use VisualSearch\Connector\Model\Config;
use VisualSearch\Connector\Model\Search\ProductPresenter;
use VisualSearch\Connector\Model\Search\ProductResolver;
use VisualSearch\Connector\Model\Shopper\ShopperResolver;

/**
 * Storefront shopping assistant: one chat turn -> the visual-search API's
 * /agent/chat -> reply text plus *Magento* product cards.
 *
 * The API's `products` are catalog SKUs; like search and recommendations they
 * are mapped back to storefront products ({@see ProductResolver}), so a card
 * is never shown for something disabled, hidden, or not in this store.
 * The shopper id comes from {@see ShopperResolver} (server side), never from
 * the browser.
 */
class AgentChat
{
    public const MAX_MESSAGE_LENGTH = 2000;

    public function __construct(
        private readonly Config $config,
        private readonly Client $client,
        private readonly ShopperResolver $shopperResolver,
        private readonly ProductResolver $productResolver,
        private readonly ProductPresenter $presenter,
        private readonly LoggerInterface $logger
    ) {
    }

    /**
     * @param array{bytes: string, mime: string, filename: string}|null $image already validated
     * @return array{reply: string, session_id: string, products: array<int, array<string, mixed>>}
     * @throws LocalizedException with a shopper-safe message if the assistant can't answer
     */
    public function send(string $message, ?string $sessionId, ?array $image): array
    {
        try {
            $response = $this->client->chat(
                $message,
                $this->shopperResolver->resolve(),
                $sessionId,
                $image,
                $this->config->getChatTimeout()
            );
        } catch (ApiException $e) {
            $this->logger->error('Shopping assistant unreachable: ' . $e->getMessage());
            throw new LocalizedException(__('The assistant is temporarily unavailable. Please try again later.'));
        }

        if (!$response->isSuccess()) {
            $this->logger->error('Shopping assistant API error: ' . $response->getErrorMessage());
            throw new LocalizedException(match ($response->getStatusCode()) {
                429 => __('The assistant is busy. Please wait a moment and try again.'),
                400, 413, 422 => __('We could not process that message or image. Please try again.'),
                default => __('The assistant is temporarily unavailable. Please try again later.'),
            });
        }

        $data = $response->getData();
        $scoreByApiSku = [];
        foreach ($data['products'] ?? [] as $product) {
            $scoreByApiSku[(string)$product['sku']] = (float)($product['score'] ?? 0.0);
        }

        return [
            'reply' => (string)($data['reply'] ?? ''),
            'session_id' => (string)($data['session_id'] ?? ''),
            'products' => array_map(
                fn (array $hit): array => $this->presenter->toArray($hit['product']),
                $this->productResolver->resolve($scoreByApiSku)
            ),
        ];
    }
}
