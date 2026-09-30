<?php
declare(strict_types=1);

namespace VisualSearch\Connector\Model\Queue;

use Magento\Framework\Serialize\Serializer\Json;
use Psr\Log\LoggerInterface;
use VisualSearch\Connector\Model\Api\ApiException;
use VisualSearch\Connector\Model\Api\Client;
use VisualSearch\Connector\Model\Config;

/**
 * Handler for the `visualsearch.event.track` queue: forwards queued shopper
 * events to POST /events.
 *
 * Run with:  bin/magento queue:consumers:start visualsearch.event.track
 * (or let Magento's cron consumers_runner pick it up).
 *
 * Events are behavioral signals, not orders: if the API is down or rejects
 * a message it is logged and dropped rather than retried forever.
 */
class EventConsumer
{
    public function __construct(
        private readonly Config $config,
        private readonly Client $client,
        private readonly Json $json,
        private readonly LoggerInterface $logger
    ) {
    }

    public function process(string $message): void
    {
        if (!$this->config->isTrackingEnabled()) {
            return;
        }

        try {
            $events = $this->json->unserialize($message)['events'] ?? [];
            if ($events === []) {
                return;
            }
            $response = $this->client->trackEvents($events);
            if (!$response->isSuccess()) {
                $this->logger->warning(sprintf(
                    'Visual search rejected %d event(s): %s',
                    count($events),
                    $response->getErrorMessage()
                ));
            }
        } catch (ApiException $e) {
            $this->logger->warning('Visual search events dropped (API unreachable): ' . $e->getMessage());
        } catch (\Throwable $e) {
            $this->logger->error('Visual search event message failed: ' . $e->getMessage(), ['exception' => $e]);
        }
    }
}
