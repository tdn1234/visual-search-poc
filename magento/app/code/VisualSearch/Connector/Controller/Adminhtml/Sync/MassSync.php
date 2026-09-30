<?php
declare(strict_types=1);

namespace VisualSearch\Connector\Controller\Adminhtml\Sync;

use Magento\Backend\App\Action;
use Magento\Backend\App\Action\Context;
use Magento\Backend\Model\View\Result\Redirect;
use Magento\Catalog\Model\ResourceModel\Product\CollectionFactory;
use Magento\Framework\App\Action\HttpPostActionInterface;
use Magento\Ui\Component\MassAction\Filter;
use VisualSearch\Connector\Model\Config;
use VisualSearch\Connector\Model\Source\SyncType;
use VisualSearch\Connector\Model\SyncService;

/**
 * Product grid mass action: queue the selected products (or, with "Select
 * All", every product matching the grid filters) for a bulk sync. Returns
 * immediately; the queue consumer does the API work.
 */
class MassSync extends Action implements HttpPostActionInterface
{
    public const ADMIN_RESOURCE = 'VisualSearch_Connector::sync';

    public function __construct(
        Context $context,
        private readonly Filter $filter,
        private readonly CollectionFactory $collectionFactory,
        private readonly Config $config,
        private readonly SyncService $syncService
    ) {
        parent::__construct($context);
    }

    public function execute(): Redirect
    {
        $redirect = $this->resultRedirectFactory->create()->setPath('catalog/product');

        if (!$this->config->isEnabled()) {
            $this->messageManager->addErrorMessage(__('The Visual Search module is disabled.'));
            return $redirect;
        }

        $productIds = $this->filter->getCollection($this->collectionFactory->create())->getAllIds();
        $queued = $this->syncService->enqueue($productIds, SyncType::BULK);

        $this->messageManager->addSuccessMessage(__(
            '%1 product(s) queued for Visual Search sync. Progress is in Catalog > Visual Search Sync Log '
            . '(requires the "visualsearch.product.sync" queue consumer to be running).',
            $queued
        ));
        return $redirect;
    }
}
