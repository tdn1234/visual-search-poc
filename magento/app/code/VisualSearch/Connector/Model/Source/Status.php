<?php
declare(strict_types=1);

namespace VisualSearch\Connector\Model\Source;

use Magento\Framework\Data\OptionSourceInterface;

/**
 * Sync log statuses.
 *
 * `queued` is the terminal state for bulk syncs: POST /products/import is
 * fire-and-forget (the API has no status endpoint), so "accepted by the
 * service's import queue" is the last thing Magento can know.
 */
class Status implements OptionSourceInterface
{
    public const PENDING = 'pending';   // waiting in the Magento queue
    public const QUEUED = 'queued';     // accepted by the API's async import queue
    public const SUCCESS = 'success';   // created in the index (single sync)
    public const SKIPPED = 'skipped';   // already exists in the index
    public const FAILED = 'failed';

    public function toOptionArray(): array
    {
        return [
            ['value' => self::PENDING, 'label' => __('Pending (Magento queue)')],
            ['value' => self::QUEUED, 'label' => __('Queued (accepted by API)')],
            ['value' => self::SUCCESS, 'label' => __('Success')],
            ['value' => self::SKIPPED, 'label' => __('Skipped')],
            ['value' => self::FAILED, 'label' => __('Failed')],
        ];
    }
}
