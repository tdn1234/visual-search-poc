<?php
declare(strict_types=1);

namespace VisualSearch\Connector\Model\ResourceModel;

use Magento\Framework\Model\ResourceModel\Db\AbstractDb;

/**
 * Sync-log table. Besides the standard model plumbing it exposes the two
 * cheap writes the sync flow needs (insert a row / patch a row by id) so
 * callers never have to load a model just to update a status.
 */
class SyncLog extends AbstractDb
{
    protected function _construct(): void
    {
        $this->_init('visual_search_sync_log', 'log_id');
    }

    /**
     * @param array<string, mixed> $data column => value
     * @return int the new log_id
     */
    public function createEntry(array $data): int
    {
        $connection = $this->getConnection();
        $connection->insert($this->getMainTable(), $data);
        return (int)$connection->lastInsertId($this->getMainTable());
    }

    /**
     * @param array<string, mixed> $data column => value
     */
    public function updateEntry(int $logId, array $data): void
    {
        $this->getConnection()->update($this->getMainTable(), $data, ['log_id = ?' => $logId]);
    }

    public function deleteOlderThan(int $days): int
    {
        $threshold = gmdate('Y-m-d H:i:s', time() - $days * 86400);
        return (int)$this->getConnection()->delete($this->getMainTable(), ['created_at < ?' => $threshold]);
    }
}
