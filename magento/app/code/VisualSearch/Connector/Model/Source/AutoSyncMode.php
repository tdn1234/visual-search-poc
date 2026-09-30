<?php
declare(strict_types=1);

namespace VisualSearch\Connector\Model\Source;

use Magento\Framework\Data\OptionSourceInterface;

class AutoSyncMode implements OptionSourceInterface
{
    public const IMMEDIATE = 'immediate';
    public const QUEUE = 'queue';

    public function toOptionArray(): array
    {
        return [
            ['value' => self::IMMEDIATE, 'label' => __('Immediate (during save)')],
            ['value' => self::QUEUE, 'label' => __('Via Magento queue')],
        ];
    }
}
