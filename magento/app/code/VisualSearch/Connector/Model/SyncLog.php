<?php
declare(strict_types=1);

namespace VisualSearch\Connector\Model;

use Magento\Framework\Model\AbstractModel;
use VisualSearch\Connector\Model\ResourceModel\SyncLog as SyncLogResource;

class SyncLog extends AbstractModel
{
    protected function _construct(): void
    {
        $this->_init(SyncLogResource::class);
    }
}
