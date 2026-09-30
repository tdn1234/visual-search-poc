<?php
declare(strict_types=1);

namespace VisualSearch\Connector\Model\Source;

use Magento\Framework\Data\OptionSourceInterface;

class SyncType implements OptionSourceInterface
{
    public const MANUAL = 'manual';
    public const AUTO = 'auto';
    public const BULK = 'bulk';

    public function toOptionArray(): array
    {
        return [
            ['value' => self::MANUAL, 'label' => __('Manual')],
            ['value' => self::AUTO, 'label' => __('On save')],
            ['value' => self::BULK, 'label' => __('Bulk')],
        ];
    }
}
