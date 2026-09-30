<?php
declare(strict_types=1);

namespace VisualSearch\Connector\Model\ResourceModel\SyncLog;

use Magento\Framework\Model\ResourceModel\Db\Collection\AbstractCollection;
use VisualSearch\Connector\Model\ResourceModel\SyncLog as SyncLogResource;
use VisualSearch\Connector\Model\SyncLog;

class Collection extends AbstractCollection
{
    protected function _construct(): void
    {
        $this->_init(SyncLog::class, SyncLogResource::class);
    }
}
