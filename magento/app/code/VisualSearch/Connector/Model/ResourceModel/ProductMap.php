<?php
declare(strict_types=1);

namespace VisualSearch\Connector\Model\ResourceModel;

use Magento\Framework\Model\ResourceModel\Db\AbstractDb;

/**
 * Magento product id <-> API sku.
 *
 * The API only accepts lowercase-slug SKUs and returns *its* sku in search
 * results, so this table is how a search hit is turned back into a Magento
 * product. Not an entity model -- just two helpers over the table.
 */
class ProductMap extends AbstractDb
{
    protected $_isPkAutoIncrement = false;

    protected function _construct(): void
    {
        $this->_init('visual_search_product_map', 'product_id');
    }

    public function upsert(int $productId, string $apiSku): void
    {
        $this->getConnection()->insertOnDuplicate(
            $this->getMainTable(),
            ['product_id' => $productId, 'api_sku' => $apiSku],
            ['api_sku']
        );
    }

    /**
     * @param string[] $apiSkus
     * @return array<string, int> api_sku => product_id
     */
    public function getProductIdsByApiSkus(array $apiSkus): array
    {
        if ($apiSkus === []) {
            return [];
        }
        $connection = $this->getConnection();
        $select = $connection->select()
            ->from($this->getMainTable(), ['api_sku', 'product_id'])
            ->where('api_sku IN (?)', $apiSkus);

        $map = [];
        foreach ($connection->fetchAll($select) as $row) {
            $map[(string)$row['api_sku']] = (int)$row['product_id'];
        }
        return $map;
    }
}
