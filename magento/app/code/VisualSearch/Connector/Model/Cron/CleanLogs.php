<?php
declare(strict_types=1);

namespace VisualSearch\Connector\Model\Cron;

use VisualSearch\Connector\Model\Config;
use VisualSearch\Connector\Model\ResourceModel\SyncLog;

class CleanLogs
{
    public function __construct(
        private readonly Config $config,
        private readonly SyncLog $syncLog
    ) {
    }

    public function execute(): void
    {
        $this->syncLog->deleteOlderThan($this->config->getLogRetentionDays());
    }
}
